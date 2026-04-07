"""
01_run_inference.py

Unified inference script for all 5 EgoBlind models.
Outputs a JSONL file with {"question_id": ..., "pred": ...} per line,
matching the format expected by the EgoBlind eval.py script.

Usage:
    python scripts/01_run_inference.py \
        --model qwen2_5_vl \
        --video_dir /scratch/jjtribb/EgoBlind_Videos/flat \
        --csv data/test_half_release.csv \
        --output outputs/predictions/qwen2_5_vl.jsonl \
        --num_frames 16

Supported --model values:
    videollama3, internvl2_5, llava_onevision, qwen2_5_vl
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# decord for fast video frame extraction
import decord
decord.bridge.set_bridge("torch")


# ─── Prompt ───────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "You are assisting a blind person who needs accurate information from video. "
    "Answer the following question based on what you can see in the video. "
    "If you cannot determine the answer from the video content, say 'I don't know'.\n"
    "Question: {question}"
)

# Chat template for lmms-lab/llava-onevision-qwen2-7b-ov.  The model upload has
# no chat_template in tokenizer_config.json; passing it explicitly to
# apply_chat_template is more reliable than setting it on the processor object.
_LLAVA_OV_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
    "{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}{{ '<image>\n' }}"
    "{% elif content['type'] == 'text' %}{{ content['text'] }}"
    "{% endif %}{% endfor %}"
    "{% endif %}"
    "{{ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


# ─── Video utilities ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, end_time: float, num_frames: int = 16) -> list:
    """
    Extract num_frames PIL Images uniformly sampled from [0, end_time] of the video.
    If end_time is invalid (<=0 or NaN), uses the full video duration.
    """
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    # Determine the end frame index
    if end_time and not np.isnan(end_time) and end_time > 0:
        end_frame = min(int(end_time * fps), total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Uniformly sample frame indices between 0 and end_frame
    indices = np.linspace(0, end_frame, num=num_frames, dtype=int)
    indices = np.clip(indices, 0, total_frames - 1)

    # Decode frames and convert to PIL
    frames_tensor = vr.get_batch(indices)  # (N, H, W, C) torch tensor
    frames = [Image.fromarray(frames_tensor[i].numpy()) for i in range(len(indices))]
    return frames


# ─── Model loaders and inference ─────────────────────────────────────────────

def load_videollama3(model_id: str):
    from transformers import AutoProcessor, AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, processor


def infer_videollama3(model, processor, frames: list, question: str, **kwargs) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question)
    # Cap resolution to avoid OOM: VideoLLaMA3 uses dynamic tiling and has no
    # built-in max_pixels setting.  At native video resolution each frame can
    # produce dozens of tiles, driving attention memory to 100+ GiB.
    # thumbnail() downsizes in-place while preserving aspect ratio.
    MAX_SIDE = 336
    frames = [f.copy() for f in frames]
    for f in frames:
        f.thumbnail((MAX_SIDE, MAX_SIDE))
    # Use every other frame (8 instead of 16) to halve visual token count.
    frames = frames[::2] if len(frames) > 8 else frames
    # When images= is provided, _load_multimodal_data (and therefore load_video /
    # ffprobe) is skipped entirely.  We still need num_frames in the conversation
    # so the Jinja template can do range(content.num_frames) without Undefined.
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "num_frames": len(frames)},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    # images expects BatchedImage: a list of videos, each video a list of PIL frames
    inputs = processor(conversation=conversation, images=[frames], return_tensors="pt")
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def load_internvl2_5(model_id: str):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def infer_internvl2_5(model, tokenizer, frames: list, question: str, **kwargs) -> str:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Each frame becomes one image tile; track how many patches per frame
    pixel_values_list = []
    num_patches_list = []
    for frame in frames:
        pv = transform(frame.convert("RGB")).unsqueeze(0).to(torch.bfloat16).cuda()
        pixel_values_list.append(pv)
        num_patches_list.append(1)

    pixel_values = torch.cat(pixel_values_list, dim=0)

    # Build prompt with one <image> token per frame
    image_tokens = "\n".join(["<image>"] * len(frames))
    prompt = PROMPT_TEMPLATE.format(question=question)
    full_prompt = f"{image_tokens}\n{prompt}"

    generation_config = {"max_new_tokens": 128, "do_sample": False}
    response = model.chat(
        tokenizer,
        pixel_values,
        full_prompt,
        generation_config,
        num_patches_list=num_patches_list,
    )
    return response.strip()


def load_llava_onevision(model_id: str):
    import json
    from huggingface_hub import hf_hub_download
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    # lmms-lab/llava-onevision-qwen2-7b-ov ships with model_type=llava in its
    # config.json, which causes transformers to mis-parse the entire nested
    # config (wrong text hidden_size, wrong vision num_heads, etc.).  Patch the
    # cached file once so every subsequent load sees the correct model_type.
    # Patch model_type in the cached config.json so transformers parses the
    # full nested config (text + vision) correctly.
    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    if cfg.get("model_type") == "llava":
        cfg["model_type"] = "llava_onevision"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    from transformers import LlavaOnevisionConfig, Qwen2Config
    config = LlavaOnevisionConfig.from_pretrained(model_id)
    # vision_config.num_attention_heads must divide embed_dim=1152 evenly.
    # The HF upload has 14 (wrong); SiGLIP-SO400M needs 16.
    config.vision_config.num_attention_heads = 16
    # The original config.json has no text_config section, so LlavaOnevisionConfig
    # defaults to hidden_size=4096 (LLaMA).  This model uses Qwen2-7B (3584).
    # Explicitly set the correct text config so image_newline and all LM weight
    # shapes match the checkpoint.
    config.text_config = Qwen2Config(
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        vocab_size=152064,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    # The lmms-lab upload has no chat_template in the tokenizer config.
    # Inject a Qwen2-style template that handles multimodal content lists.
    if not getattr(processor.tokenizer, "chat_template", None):
        processor.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
            "{% endif %}"
            "{{'<|im_start|>' + message['role'] + '\n'}}"
            "{% if message['content'] is string %}{{ message['content'] }}"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'image' %}{{ '<image>\n' }}"
            "{% elif content['type'] == 'text' %}{{ content['text'] }}"
            "{% endif %}{% endfor %}"
            "{% endif %}"
            "{{ '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return model, processor


def infer_llava_onevision(model, processor, frames: list, question: str, **kwargs) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question)
    image_content = [{"type": "image"} for _ in frames]
    conversation = [
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": prompt}],
        }
    ]
    # Pass template explicitly — more reliable than relying on processor.chat_template
    # or processor.tokenizer.chat_template, which are unset in the lmms-lab upload.
    text = processor.apply_chat_template(
        conversation,
        chat_template=_LLAVA_OV_CHAT_TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=text, images=frames, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.decode(generated[0], skip_special_tokens=True).strip()


def load_qwen2_5_vl(model_id: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    # max_pixels caps each frame's visual token count to avoid OOM.
    # Default (1280*28*28 ≈ 1M px/frame) × 16 frames exhausts GPU memory;
    # 256*28*28 ≈ 200K px/frame keeps the attention computation tractable.
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=True,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, processor


def infer_qwen2_5_vl(model, processor, frames: list, question: str, **kwargs) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question)
    # One {"type": "image"} dict per frame in the message content
    image_content = [{"type": "image", "image": frame} for frame in frames]
    messages = [
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=frames,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.decode(generated[0], skip_special_tokens=True).strip()


# ─── Model registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "videollama3": {
        "model_id": "DAMO-NLP-SG/VideoLLaMA3-7B",
        "load_fn": load_videollama3,
        "infer_fn": infer_videollama3,
    },
    "internvl2_5": {
        "model_id": "OpenGVLab/InternVL2_5-8B",
        "load_fn": load_internvl2_5,
        "infer_fn": infer_internvl2_5,
    },
    "llava_onevision": {
        "model_id": "lmms-lab/llava-onevision-qwen2-7b-ov",
        "load_fn": load_llava_onevision,
        "infer_fn": infer_llava_onevision,
    },
    "qwen2_5_vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "load_fn": load_qwen2_5_vl,
        "infer_fn": infer_qwen2_5_vl,
    },
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    registry = MODEL_REGISTRY[args.model]
    model_id = args.model_id or registry["model_id"]
    load_fn = registry["load_fn"]
    infer_fn = registry["infer_fn"]

    # Load dataset
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} questions from {args.csv}")

    # Resume support: skip already-processed question_ids
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    processed_ids.add(obj["question_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(processed_ids)} questions already processed, skipping them.")

    remaining = df[~df["question_id"].isin(processed_ids)]
    print(f"Questions to process: {len(remaining)}")

    if len(remaining) == 0:
        print("Nothing to do — all questions already processed.")
        return

    # Load model
    print(f"Loading model: {model_id}")
    model, processor = load_fn(model_id)
    print("Model loaded.")

    # Inference loop
    video_dir = Path(args.video_dir)
    errors = 0

    with open(output_path, "a") as out_f:
        for i, row in enumerate(remaining.itertuples(index=False)):
            # Resolve video path: video_name is an int like 923 → 00923.mp4
            video_filename = f"{int(row.video_name):05d}.mp4"
            video_path = video_dir / video_filename

            try:
                if not video_path.exists():
                    raise FileNotFoundError(f"Video not found: {video_path}")

                # Extract frames up to the question timestamp
                end_time = getattr(row, "start-time/s", None)
                frames = extract_frames(str(video_path), end_time, num_frames=args.num_frames)

                # Run inference
                pred = infer_fn(model, processor, frames, row.question)

                out_f.write(json.dumps({"question_id": row.question_id, "pred": pred}) + "\n")
                out_f.flush()

            except Exception as e:
                errors += 1
                err_msg = f"ERROR: {type(e).__name__}: {e}"
                print(f"[{i+1}/{len(remaining)}] ERROR on {row.question_id}: {err_msg}")
                traceback.print_exc()
                # Write error entry so resume logic can skip it; eval.py will penalize it as wrong
                out_f.write(json.dumps({"question_id": row.question_id, "pred": "I don't know", "error": err_msg}) + "\n")
                out_f.flush()
                continue

            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(remaining)}] Processed {i+1} questions ({errors} errors so far)")

    print(f"\nDone. Total errors: {errors}")
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()),
                        help="Which model to run")
    parser.add_argument("--model_id", default=None,
                        help="Override HuggingFace model ID (optional)")
    parser.add_argument("--video_dir", required=True,
                        help="Path to flat directory containing all .mp4 files")
    parser.add_argument("--csv", default="data/test_half_release.csv",
                        help="Path to test_half_release.csv")
    parser.add_argument("--output", required=True,
                        help="Output .jsonl file path")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames to sample per video (default: 16)")
    args = parser.parse_args()
    main(args)
