"""
preload.py

Pre-downloads all 5 EgoBlind model weights and processors to the HuggingFace cache.
Run this on an interactive node with internet access before submitting SLURM batch jobs,
since compute nodes may not have outbound internet.

Usage:
    source scripts/set_cache_dirs.sh
    python scripts/preload.py

Each model is downloaded independently; a failure on one does not stop the rest.
A summary of successes and failures is printed at the end.
"""

from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

MODELS = [
    {
        "name": "VideoLLaMA3",
        "model_id": "DAMO-NLP-SG/VideoLLaMA3-7B",
        "model_cls": AutoModel,
        "processor_cls": AutoProcessor,
        "trust_remote_code": True,
    },
    {
        "name": "InternVL2_5-8B",
        "model_id": "OpenGVLab/InternVL2_5-8B",
        "model_cls": AutoModel,
        "processor_cls": AutoTokenizer,
        "trust_remote_code": True,
    },
    {
        "name": "LLaVA-OneVision",
        "model_id": "lmms-lab/llava-onevision-qwen2-7b-ov",
        "model_cls": LlavaOnevisionForConditionalGeneration,
        "processor_cls": AutoProcessor,
        "trust_remote_code": False,
    },
    {
        "name": "Qwen2.5-VL",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_cls": Qwen2_5_VLForConditionalGeneration,
        "processor_cls": AutoProcessor,
        "trust_remote_code": False,
    },
    {
        "name": "MiniCPM-V-2_6",
        "model_id": "openbmb/MiniCPM-V-2_6",
        "model_cls": AutoModel,
        "processor_cls": AutoTokenizer,
        "trust_remote_code": True,
    },
]


def download_model(entry: dict) -> None:
    name = entry["name"]
    model_id = entry["model_id"]
    trust = entry["trust_remote_code"]

    print(f"\n[{name}] Downloading processor from {model_id} ...")
    entry["processor_cls"].from_pretrained(model_id, trust_remote_code=trust)
    print(f"[{name}] Processor done. Downloading model weights ...")
    entry["model_cls"].from_pretrained(model_id, trust_remote_code=trust)
    print(f"[{name}] Model done.")


def main():
    results = {}

    for entry in MODELS:
        name = entry["name"]
        try:
            download_model(entry)
            results[name] = "OK"
        except Exception as e:
            results[name] = f"FAILED — {type(e).__name__}: {e}"
            print(f"[{name}] ERROR: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("Download summary:")
    for name, status in results.items():
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {name}: {status}")

    failures = [n for n, s in results.items() if s != "OK"]
    if failures:
        print(f"\n{len(failures)} model(s) failed. Re-run to retry, or check the error above.")
    else:
        print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    main()
