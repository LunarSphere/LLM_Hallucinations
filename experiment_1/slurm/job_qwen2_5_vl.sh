#!/bin/bash
#SBATCH --job-name=egoblind_qwen2_5
#SBATCH --nodes 1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu 8gb
#SBATCH --gpus-per-task h200:1
#SBATCH --time 12:00:00   
#SBATCH --output=logs/qwen2_5_%j.out
#SBATCH --error=logs/qwen2_5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jjtribb@clemson.edu

module load anaconda3/2023.09
source activate exp1

source /home/jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh

cd /home/jjtribb/LLM_Hallucinations/experiment_1

mkdir -p logs outputs/predictions

python scripts/01_run_inference.py \
    --model qwen2_5_vl \
    --video_dir /scratch/jjtribb/EgoBlind_Videos/flat \
    --csv data/test_half_release.csv \
    --output outputs/predictions/qwen2_5_vl.jsonl \
    --num_frames 16
