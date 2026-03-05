#!/bin/bash
#SBATCH --account=****
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100'
#SBATCH --time=47:59:59
#SBATCH --mem=128GB
#SBATCH --array=0-8
#SBATCH --job-name=****
#SBATCH --output=slurm_logs/stepwise_%A_%a.out
#SBATCH --error=slurm_logs/stepwise_%A_%a.err


BASE_DIR=$(pwd)
MARK_ID=$(basename "$BASE_DIR" | grep -oP 'Iteration\K[0-9]+')
MARK_ID=${MARK_ID:-29}

EXPS=("E1" "E2" "E3" "E4" "E5" "E6" "E7" "E8" "E9")
EXP_NAME=${EXPS[$SLURM_ARRAY_TASK_ID]}

export HF_HOME="*/hf_cache"
export TRANSFORMERS_CACHE="*/hf_cache"
export HF_DATASETS_CACHE="*/hf_cache"
export VLLM_CACHE_ROOT="*/vllm_cache"

module purge
if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

OVERLAY_PATH="*/pytorch_${**}.ext3:ro"

singularity exec $nv \
--overlay $OVERLAY_PATH \
*/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "export HF_TOKEN='*************************'; \
source */env.sh; \
cd ${BASE_DIR}; \
mkdir -p output; \
python3 stepwise_runner.py $EXP_NAME 2>&1 | tee output/stepwise_${EXP_NAME}.log"
