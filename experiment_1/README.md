# Experiment 1: EgoBlind Hallucination Evaluation

Evaluates 5 open-source MLLMs on the EgoBlind benchmark to measure how often they
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

### 3. Create the environment

The conda solver hangs on Palmetto when resolving pytorch/nvidia channels. Use a bare
conda env for Python only, then install everything via pip:

```bash
conda create -n egoblind_exp1 python=3.10 -y
conda activate egoblind_exp1
pip install -r requirements.txt
```

InternVL2.5 requires `flash-attn`. Building from source fails on Palmetto due to
GCC/CUDA version constraints. Install the precompiled wheel instead:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 4. Pre-download all 5 models (do this on an interactive node with internet access)

```bash
srun --pty --partition=work1 --mem=32G --time=02:00:00 bash
conda activate exp1
source /home/jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh
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

### Step 2 — Submit all 5 inference jobs (can run in parallel)

```bash
cd /LLM_Hallucinations/experiment_1

J1=$(sbatch --parsable slurm/job_videollama3.sh)
J2=$(sbatch --parsable slurm/job_internvl2_5.sh)
J3=$(sbatch --parsable slurm/job_llava_onevision.sh)
J4=$(sbatch --parsable slurm/job_qwen2_5_vl.sh)

echo "Submitted jobs: $J1 $J2 $J3 $J4"
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
sbatch --dependency=afterok:${J1}:${J2}:${J3}:${J4} slurm/job_evaluate.sh
```

This runs `eval.py` (the official EgoBlind evaluator) for all 5 models, then
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

- All models use **16 uniformly sampled frames** up to the question timestamp (`start-time/s`)
- The prompt includes blind-user framing: *"You are assisting a blind person..."*
- `eval.py` requires an OpenAI API key and takes ~18 minutes per model
- InternVL2.5 requests 80G RAM due to higher memory overhead
- Set `TRANSFORMERS_OFFLINE=1` in SLURM scripts after pre-downloading models
