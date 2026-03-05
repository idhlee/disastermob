#!/bin/bash
#SBATCH --account=****
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100'
#SBATCH --time=47:59:59
#SBATCH --mem=128GB
#SBATCH --job-name=****
#SBATCH --output=slurm_logs/lora_train_%j.out
#SBATCH --error=slurm_logs/lora_train_%j.err

BASE_DIR=$(pwd)

mkdir -p ${BASE_DIR}/slurm_logs

export HF_HOME="*/hf_cache"
export HF_DATASETS_CACHE="*/hf_datasets"
export TRANSFORMERS_CACHE="*/transformers_cache"
export TORCH_HOME="*/torch_cache"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Base directory: $BASE_DIR"
echo "========================================"

module purge
if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

OVERLAY_PATH="*/pytorch_${**}.ext3:ro"

singularity exec $nv \
  --overlay $OVERLAY_PATH \
  */singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "export HF_TOKEN='*************************'; \
  source */env.sh; \
  cd */LoRA; \
  python3 LoRA_training.py 2>&1 | tee output/lora_training_${SLURM_JOB_ID}.log"

echo "========================================"
echo "Job finished at: $(date)"
