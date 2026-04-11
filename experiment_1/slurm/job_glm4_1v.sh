#!/bin/bash
#SBATCH --job-name=egoblind_glm4_1v
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu 8gb
#SBATCH --gpus-per-task h200:1
#SBATCH --time 04:00:00
#SBATCH --output=logs/glm4_1v_%j.out
#SBATCH --error=logs/glm4_1v_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jjtribb@clemson.edu

module load anaconda3/2023.09
source activate exp1_glm

source /home/jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh

cd /home/jjtribb/LLM_Hallucinations/experiment_1

mkdir -p logs outputs/predictions

python scripts/01_run_inference.py \
    --model glm4_1v \
    --video_dir /scratch/jjtribb/EgoBlind_Videos/flat \
    --csv data/test_half_release.csv \
    --output outputs/predictions/glm4_1v.jsonl \
    --num_frames 16
