#!/bin/bash
#SBATCH --job-name=egoblind_videollama3
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/videollama3_%j.out
#SBATCH --error=logs/videollama3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jjtribb@clemson.edu

module load anaconda3/2023.09
source activate egoblind_exp1

source /jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh

cd /jjtribb/LLM_Hallucinations/experiment_1

mkdir -p logs outputs/predictions

python scripts/01_run_inference.py \
    --model videollama3 \
    --video_dir /scratch/jjtribb/EgoBlind_Videos/flat \
    --csv data/test_half_release.csv \
    --output outputs/predictions/videollama3.jsonl \
    --num_frames 16
