# Experiment 1: EgoBlind Hallucination Evaluation

Evaluates 6 open-source MLLMs on the EgoBlind benchmark to measure how often they
hallucinate answers on unanswerable video QA questions (where the ground truth is "I don't know").

---

## Setup on Palmetto HPC

### 1. Clone this project and the EgoBlind repo

```bash
# Repos live in the small home dir (/jjtribb); caches go to scratch (/scratch/jjtribb)
cd /jjtribb/LLM_Hallucinations/experiment_1
git clone https://github.com/doc-doc/EgoBlind.git _egoblind_repo
cp _egoblind_repo/eval.py .
```

### 2. Copy the dataset CSV

```bash
cp /path/to/test_half_release.csv data/test_half_release.csv
```

### 3. Create the environments

The conda solver hangs on Palmetto when resolving pytorch/nvidia channels. Use a bare
conda env for Python only, then install everything via pip.

**Primary environment** (VideoLLaMA3, InternVL2.5, LLaVA-OneVision, Qwen2.5-VL):
```bash
conda create -n exp1 python=3.10 -y
conda activate exp1
pip install -r requirements.txt
```

**Qwen3-VL / Gemma 4 environment** (requires `transformers>=4.57.0`, separate from the others):
```bash
conda create -n exp1_qwen3 python=3.10 -y
conda activate exp1_qwen3
pip install -r requirements_qwen3_vl.txt
```

InternVL2.5 requires `flash-attn`. Building from source fails on Palmetto due to
GCC/CUDA version constraints. Install the precompiled wheel instead (into `exp1`):

```bash
conda activate exp1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 4. Pre-download all models (do this on an interactive node with internet access)

`preload.py` downloads all models including Qwen3-VL and Gemma 4; you only need `exp1` active since
it uses `snapshot_download` (no model class instantiation).

> **Gemma 4 is a gated model.** Before running `preload.py`, log in to HuggingFace and accept
> the model terms at `huggingface.co/google/gemma-4-31B-it`, then run `huggingface-cli login`.

```bash
srun --pty --partition=work1 --mem=32G --time=02:00:00 bash
conda activate exp1
source /home/jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE   # set_cache_dirs sets these to 1; unset here so downloads work
cd /home/jjtribb/LLM_Hallucinations/experiment_1
python scripts/preload.py
exit
```

---

## Running the Experiment

### Step 1 — Consolidate videos into a flat folder

```bash
python scripts/00_consolidate_videos.py \
    --src /scratch/jjtribb/EgoBlind_Videos \
    --dst /scratch/jjtribb/EgoBlind_Videos/flat \
    --csv data/test_half_release.csv

# Preview without moving files:
python scripts/00_consolidate_videos.py --dry_run
```

### Step 2 — Submit all inference jobs (can run in parallel)

```bash
cd /LLM_Hallucinations/experiment_1

J1=$(sbatch --parsable slurm/job_videollama3.sh)
J2=$(sbatch --parsable slurm/job_internvl2_5.sh)
J3=$(sbatch --parsable slurm/job_llava_onevision.sh)
J4=$(sbatch --parsable slurm/job_qwen2_5_vl.sh)
J5=$(sbatch --parsable slurm/job_qwen3_vl.sh)
J6=$(sbatch --parsable slurm/job_gemma4.sh)

echo "Submitted jobs: $J1 $J2 $J3 $J4 $J5 $J6"
```

Monitor progress:
```bash
squeue -u jjtribb
tail -f logs/qwen2_5_vl_<jobid>.out
```

### Step 3 — Submit evaluation job (runs after all inference jobs finish)

```bash
# Replace your OpenAI API key in slurm/job_evaluate.sh first!
# Then:
sbatch --dependency=afterok:${J1}:${J2}:${J3}:${J4}:${J5}:${J6} slurm/job_evaluate.sh
```

This runs `eval.py` (the official EgoBlind evaluator) for all 6 models, then
`02_analyze_hallucination.py` to compute IDK Rate and Hallucination Rate.

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/predictions/{model}.jsonl` | Raw model predictions |
| `outputs/evaluations/metrics_{model}.json` | GPT Score by question type (from eval.py) |
| `outputs/evaluations/results_{model}.json` | Per-question eval results (from eval.py) |
| `outputs/final/hallucination_summary.csv` | IDK Rate + Hallucination Rate per model |
| `outputs/final/hallucination_by_type.csv` | Hallucination breakdown by question type |

---

## Resuming Interrupted Jobs

If a SLURM job times out, just resubmit the same script. The inference script
checks which `question_id`s are already in the output JSONL and skips them.

---

## Notes

- All models use **16 uniformly sampled frames** up to the question timestamp (`start-time/s`),
  except Gemma 4 which subsamples to 8 frames internally (model weights occupy ~62 GB on H200,
  leaving limited headroom for activations)
- The prompt includes blind-user framing: *"You are assisting a blind person..."*
- `eval.py` requires an OpenAI API key and takes ~18 minutes per model
- InternVL2.5 requests 80G RAM due to higher memory overhead
- Set `TRANSFORMERS_OFFLINE=1` in SLURM scripts after pre-downloading models
- **Qwen3-VL and Gemma 4** share the `exp1_qwen3` conda env (`requirements_qwen3_vl.txt`);
  requires `transformers>=4.57.0` and `qwen-vl-utils>=0.0.14` which conflict with the
  `transformers>=4.49.0` baseline used by the other four models
- **Gemma 4 is gated** — accept terms at `huggingface.co/google/gemma-4-31B-it` and run
  `huggingface-cli login` before pre-downloading
