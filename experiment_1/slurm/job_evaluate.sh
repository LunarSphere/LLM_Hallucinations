#!/bin/bash
#SBATCH --job-name=egoblind_eval
#SBATCH --partition=work1           # CPU partition — no GPU needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jjtribb@clemson.edu

# Submit this job with a dependency on all 5 inference jobs:
#   sbatch --dependency=afterok:<j1>:<j2>:<j3>:<j4>:<j5> slurm/job_evaluate.sh

module load anaconda3/2023.09
source activate egoblind_exp1

source /jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh
export OPENAI_API_KEY="<your_openai_api_key_here>"

cd /jjtribb/LLM_Hallucinations/experiment_1

mkdir -p outputs/evaluations

# Run eval.py (from EgoBlind repo, cloned into experiment_1/) for each model
for model in videollama3 internvl2_5 llava_onevision qwen2_5_vl minicpm_v; do
    pred_file="outputs/predictions/${model}.jsonl"
    if [ -f "$pred_file" ]; then
        echo "Evaluating $model..."
        python eval.py \
            --pred_path "$pred_file" \
            --test_path data/test_half_release.csv \
            --output_dir outputs/evaluations \
            --model_name "$model"
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
