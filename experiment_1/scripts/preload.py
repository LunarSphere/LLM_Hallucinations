"""
preload.py

Pre-downloads all EgoBlind model weights and processors to the HuggingFace cache.
Run this on an interactive node with internet access before submitting SLURM batch jobs,
since compute nodes may not have outbound internet.

Uses snapshot_download to pull files only — no model class instantiation — so this
works regardless of local environment quirks (transformers version, CUDA, etc.).

Usage:
    source scripts/set_cache_dirs.sh
    python scripts/preload.py
"""

from huggingface_hub import snapshot_download

MODELS = [
    {"name": "VideoLLaMA3",      "model_id": "DAMO-NLP-SG/VideoLLaMA3-7B"},
    {"name": "InternVL2_5-8B",   "model_id": "OpenGVLab/InternVL2_5-8B"},
    {"name": "InternVL3_5-8B-HF", "model_id": "OpenGVLab/InternVL3_5-8B-HF"},
    {"name": "LLaVA-OneVision",  "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf"},
    {"name": "Qwen2.5-VL",       "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    {"name": "Qwen3-VL",         "model_id": "Qwen/Qwen3-VL-8B-Instruct"},
    {"name": "Gemma4-26B-A4B-it",       "model_id": "google/gemma-4-26B-A4B-it"},
    {"name": "VideoChat-R1-thinking-7B", "model_id": "OpenGVLab/VideoChat-R1-thinking_7B"},
]


def download_model(entry: dict) -> None:
    name = entry["name"]
    model_id = entry["model_id"]
    print(f"\n[{name}] Downloading {model_id} ...")
    snapshot_download(model_id)
    print(f"[{name}] Done.")


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
