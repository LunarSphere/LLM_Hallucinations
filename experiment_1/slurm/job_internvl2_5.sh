#!/bin/bash
#SBATCH --job-name=egoblind_internvl
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/internvl2_5_%j.out
#SBATCH --error=logs/internvl2_5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jjtribb@clemson.edu

module load anaconda3/2023.09
source activate egoblind_exp1

source /jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh

cd /jjtribb/LLM_Hallucinations/experiment_1

mkdir -p logs outputs/predictions

python scripts/01_run_inference.py \
    --model internvl2_5 \
    --video_dir /scratch/jjtribb/EgoBlind_Videos/flat \
    --csv data/test_half_release.csv \
    --output outputs/predictions/internvl2_5.jsonl \
    --num_frames 16
