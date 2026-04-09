# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project investigating hallucination behavior in open-source Multimodal Large Language Models (MLLMs) on video question answering tasks. The project uses the [EgoBlind benchmark](https://arxiv.org/pdf/2503.08221) — first-person video QA filmed by blind/visually impaired individuals — to measure whether models hallucinate answers on questions that humans marked as unanswerable ("I don't know").

**All code runs on Clemson's [Palmetto HPC cluster](https://docs.rcd.clemson.edu/palmetto/). Do not execute Python scripts or create conda environments locally.**

## Repository Structure

Experiments live in numbered subdirectories (`experiment_1/`, `experiment_2/`, etc.), each self-contained with its own:
- `scripts/` — Python scripts (numbered for execution order)
- `slurm/` — SLURM job submission scripts
- `environment.yaml` — conda environment definition
- `README.md` — setup and run instructions
- `data/`, `outputs/`, `logs/` — gitignored runtime directories

## Experiment 1 Pipeline

The pipeline runs in three steps, each a separate script:

**Step 0 — Video consolidation** (one-time setup):
```bash
python scripts/00_consolidate_videos.py \
    --src /scratch/jjtribb/EgoBlind_Videos \
    --dst /scratch/jjtribb/EgoBlind_Videos/flat \
    --csv data/test_half_release.csv
# Add --dry_run to preview without moving files
```

**Step 1 — Inference** (submit as SLURM jobs, can run in parallel):
```bash
J1=$(sbatch --parsable slurm/job_videollama3.sh)
J2=$(sbatch --parsable slurm/job_internvl2_5.sh)
J3=$(sbatch --parsable slurm/job_llava_onevision.sh)
J4=$(sbatch --parsable slurm/job_qwen2_5_vl.sh)
```
Or run a single model manually:
```bash
python scripts/01_run_inference.py \
    --model qwen2_5_vl \
    --video_dir /scratch/jjtribb/EgoBlind_Videos/flat \
    --csv data/test_half_release.csv \
    --output outputs/predictions/qwen2_5_vl.jsonl
```

**Step 2 — Evaluation + analysis** (depends on all inference jobs):
```bash
# Run eval.py (from EgoBlind repo) then analyze hallucination metrics
sbatch --dependency=afterok:${J1}:${J2}:${J3}:${J4} slurm/job_evaluate.sh

# Or run analysis directly after eval.py has been run:
python scripts/02_analyze_hallucination.py \
    --results_dir outputs/evaluations \
    --pred_dir outputs/predictions \
    --csv data/test_half_release.csv
```

## Key Architecture Details

**`01_run_inference.py`** — Unified script for all 5 models via a `MODEL_REGISTRY` dict. Each entry has a `load_fn` and `infer_fn`. The script supports **resume**: already-processed `question_id`s in the output JSONL are skipped, so interrupted SLURM jobs can be resubmitted without re-running completed questions. Frames are extracted using `decord` up to the question timestamp (`start-time/s` column).

**`02_analyze_hallucination.py`** — Reads per-question eval results from `eval.py` (official EgoBlind evaluator using GPT-4o mini as judge) and computes:
- **IDK Rate**: % of unanswerable questions where the model said "I don't know"
- **Hallucination Rate**: % of unanswerable questions where the model gave a confident (wrong) answer
- Per question-type breakdown (Navigation, Safety, Tool Use, etc.)

**Unanswerable detection**: `is_idk_response()` uses `IDK_REGEX` — a set of patterns matching "I don't know", "cannot determine", "not visible", etc. — applied to both GT answers and model predictions.

**`eval.py`** is not in this repo; clone it from `https://github.com/doc-doc/EgoBlind` into `experiment_1/_egoblind_repo/` and copy to `experiment_1/eval.py`.

## HPC Setup

**Environment** — the conda solver hangs on Palmetto. Use a bare conda env for Python only, then `pip install -r requirements.txt` for everything else:
```bash
conda create -n egoblind_exp1 python=3.10 -y
conda activate egoblind_exp1
pip install -r requirements.txt
```

**flash-attn** — build-from-source fails on Palmetto (CUDA 11.8 / GCC version mismatch). Use the precompiled wheel:
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

**Cache directories** (set before downloading models or running inference):
```bash
source /jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh
```

**Pre-download models** on an interactive node (compute nodes may lack outbound internet). `preload.py` uses the correct loader class per model and reports a pass/fail summary:
```bash
python scripts/preload.py
```
Set `TRANSFORMERS_OFFLINE=1` in SLURM scripts after downloading.

**Memory requirements**: InternVL2.5 requests 80G RAM; others use less.

## Adding a New Model to Experiment 1

Follow these steps in order. Each step maps to one file change. Always browse the HuggingFace model card before coding to verify the correct API, tokenizer flags, and transformers version requirements.

### Step 1 — `01_run_inference.py`: write `load_` and `infer_` functions

Add `load_<model_key>(model_id: str) -> (model, processor)` and `infer_<model_key>(model, processor, frames: list, question: str, **kwargs) -> str` near the other model implementations (before the `MODEL_REGISTRY` block). Base the new functions on the closest existing model:

| Architecture | Template to copy | Notes |
|---|---|---|
| InternVL family | `load_internvl3_5` / `infer_internvl3_5` | `AutoModel` + `model.chat()`, `Frame{i}: <image>` tokens, `use_flash_attn=False` in `exp1_qwen3` |
| Qwen-VL family | `load_qwen2_5_vl` / `infer_qwen2_5_vl` | `AutoProcessor` + standard HF `generate()` |
| Reasoning/CoT | `infer_videochat_r1` | `<answer>` tag extraction, `max_new_tokens=512` |

Update the module docstring at the top of `01_run_inference.py` to list the new `--model` key.

### Step 2 — `01_run_inference.py`: add to `MODEL_REGISTRY`

```python
"model_key": {
    "model_id": "org/repo-name",
    "load_fn": load_model_key,
    "infer_fn": infer_model_key,
},
```

### Step 3 — `preload.py`: add to `MODELS` list

```python
{"name": "DisplayName", "model_id": "org/repo-name"},
```

### Step 4 — Create SLURM script `slurm/job_<model_key>.sh`

Copy the closest existing script and change: `--job-name`, `--output`, `--error`, `--gpus-per-task` (h100 for older/smaller; h200 for newer/larger), `source activate` env, `--model` arg, and `--output` path.

### Step 5 — Update `README.md`

- Add `J<N>=$(sbatch --parsable slurm/job_<model_key>.sh)` to the submission block and update the `echo` line
- Update `--dependency=afterok:...` to include `${J<N>}`
- Add a bullet to Notes documenting environment and any special setup

### Step 6 — Determine conda environment

| Situation | Environment |
|---|---|
| Needs `transformers>=4.52` | `exp1_qwen3` (`requirements_qwen3_vl.txt`) — covers InternVL3.5, Qwen3-VL, Gemma4 |
| Needs flash-attn + older transformers | `exp1` (`requirements.txt`) + precompiled wheel |
| Conflicting dependency | New `exp1_<name>` env + new requirements file |

### Common gotchas

- **flash-attn in `exp1_qwen3`**: not installed. For InternVL models pass `use_flash_attn=False` (custom model kwarg). For HF-native models use `attn_implementation="eager"`.
- **`low_cpu_mem_usage=True` + `device_map="auto"`**: do NOT combine these for InternVL models. The combination initializes tensors as meta tensors; InternVL's vision encoder `__init__` calls `.item()` during drop-path-rate setup, which crashes with `RuntimeError: Tensor.item() cannot be called on meta tensors`. Omit `low_cpu_mem_usage` and let `device_map="auto"` handle placement alone.
- **`torch_dtype` deprecated**: use `dtype=` instead in `from_pretrained()`.
- **Mistral regex warning (false positive)**: InternVL3.5 and other non-Mistral models saved with `transformers>=4.57.3` trigger a "fix_mistral_regex" warning — this is a false positive from overly broad detection logic. Do NOT pass `fix_mistral_regex=True` when using `use_fast=False`: `_patch_mistral_regex` accesses `.backend_tokenizer` which only exists on fast tokenizers and crashes on slow ones. Ignore the warning; tokenization is unaffected.
- **Gated models**: run `huggingface-cli login` and accept terms before `preload.py`.
- **`token_type_ids`**: Qwen3-VL's processor injects these; `model.generate()` rejects them — pop before generate: `inputs.pop("token_type_ids", None)`.
- **Reasoning models**: use `max_new_tokens=512` to avoid truncating the `<think>` block before `<answer>` appears.
- **OOM on h100**: switch to h200 in the SLURM script.
- **`use_fast=False`**: required by InternVL tokenizers (SentencePiece-based).

## Adding New Experiments

Create a new `experiment_N/` directory with the same structure (`scripts/`, `slurm/`, `environment.yaml`, `README.md`). Add `experiment_N/outputs/`, `experiment_N/logs/`, and `experiment_N/data/` to `.gitignore`.
