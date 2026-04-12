#!/bin/bash
#SBATCH --job-name=egoblind_eval
#SBATCH --partition=work1           # CPU partition — no GPU needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jjtribb@clemson.edu

# Submit this job with a dependency on all inference jobs:
#   sbatch --dependency=afterok:<j1>:<j2>:<j3>:<j4>:<j5>:<j6>:<j7>:<j8>:<j9> slurm/job_evaluate.sh

module load anaconda3/2023.09
source activate exp1

source /home/jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh
export OPENAI_API_KEY="<your_openai_api_key_here>"

cd /home/jjtribb/LLM_Hallucinations/experiment_1

mkdir -p outputs/evaluations

# Run eval.py (from EgoBlind repo, cloned into experiment_1/) for each model
# eval.py only accepts --pred_path and --test_path; it derives model_name from the
# prediction filename and writes result_{model}.json relative to CWD.
# Run from outputs/evaluations/ so output lands there.
for model in videollama3 internvl2_5 internvl3_5 llava_onevision qwen2_5_vl videochat_r1 qwen3_vl gemma4 glm4_1v; do
    pred_file="/home/jjtribb/LLM_Hallucinations/experiment_1/outputs/predictions/${model}.jsonl"
    if [ -f "$pred_file" ]; then
        echo "Evaluating $model..."
        (cd outputs/evaluations && python ../../eval.py \
            --pred_path "$pred_file" \
            --test_path /home/jjtribb/LLM_Hallucinations/experiment_1/data/test_half_release.csv)
    else
        echo "WARNING: Predictions file not found for $model: $pred_file"
    fi
done

echo "Evaluation complete. Running hallucination analysis..."

python scripts/02_analyze_hallucination.py \
    --results_dir outputs/evaluations \
    --pred_dir outputs/predictions \
    --csv data/test_half_release.csv \
    --output_dir outputs/final
