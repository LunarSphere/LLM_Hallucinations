#!/bin/bash
# set_cache_dirs.sh
#
# Source this file in every SLURM job script to redirect all model/library
# caches away from the small home directory (/jjtribb) to scratch storage.
#
# Usage (in SLURM scripts):
#   source /jjtribb/LLM_Hallucinations/experiment_1/scripts/set_cache_dirs.sh

SCRATCH=/scratch/jjtribb

# ── HuggingFace ────────────────────────────────────────────────────────────────
export HF_HOME=${SCRATCH}/hf_cache               # models, tokenizers, datasets
export HF_DATASETS_CACHE=${SCRATCH}/hf_cache/datasets

# ── PyTorch ────────────────────────────────────────────────────────────────────
export TORCH_HOME=${SCRATCH}/torch_cache          # torch hub models

# ── pip / misc ─────────────────────────────────────────────────────────────────
export PIP_CACHE_DIR=${SCRATCH}/pip_cache
export TRITON_CACHE_DIR=${SCRATCH}/triton_cache   # triton kernel cache

# ── Behaviour flags ────────────────────────────────────────────────────────────
# Set to 1 after pre-downloading all models so jobs never hit the internet
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Create cache directories if they don't exist yet
mkdir -p \
    ${HF_HOME}/hub \
    ${HF_DATASETS_CACHE} \
    ${TORCH_HOME} \
    ${PIP_CACHE_DIR} \
    ${TRITON_CACHE_DIR}

echo "[cache] All caches routed to ${SCRATCH}"
