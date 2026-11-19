
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:59
#SBATCH --mem=128G
#SBATCH --array=1-5%5
#SBATCH --job-name=ft_marshall
#SBATCH --output=ft_%A_%a.out
#SBATCH --error=ft_%A_%a.err

set -euo pipefail

# ==================================================
# 1. Project Configuration
# ==================================================

BASE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

CONFIG_PATH="${CONFIG_PATH:-${BASE_DIR}/inference_config.yaml}"
RUNNER="${BASE_DIR}/stepwise_runner.py"

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-singularity}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"

OVERLAY_FILE="${OVERLAY_FILE:-}"
CONTAINER_ENV_SCRIPT="${CONTAINER_ENV_SCRIPT:-}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR}"
ADAPTER_ROOT="${ADAPTER_ROOT:?Set ADAPTER_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/outputs}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_DIR}/cache}"

# ==================================================
# 2. Fold Configuration
# ==================================================

FOLD="${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID}"

DATA_PATH="${DATA_DIR}/fold${FOLD}_test.csv"
ADAPTER_PATH_TEMPLATE="${ADAPTER_ROOT}/fold{fold}/final_adapter"
ADAPTER_PATH="${ADAPTER_ROOT}/fold${FOLD}/final_adapter"

ITERATION=1
BASE_SEED=6000
VLLM_SEED="$((BASE_SEED + ITERATION))"

[[ "$FOLD" =~ ^[1-5]$ ]] || {
    echo "Invalid fold: $FOLD" >&2
    exit 1
}

# ==================================================
# 3. Validate Paths
# ==================================================

for path in \
    "${CONFIG_PATH}" \
    "${RUNNER}" \
    "${BASE_DIR}/model.py" \
    "${BASE_DIR}/agent.py" \
    "${MODEL_PATH}/config.json" \
    "${DATA_PATH}" \
    "${ADAPTER_PATH}/adapter_config.json" \
    "${CONTAINER_IMAGE}"; do

    [[ -e "${path}" ]] || {
        echo "Required path not found: ${path}" >&2
        exit 1
    }
done

if [[ -n "${OVERLAY_FILE}" && ! -f "${OVERLAY_FILE}" ]]; then
    echo "Overlay not found: ${OVERLAY_FILE}" >&2
    exit 1
fi

command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1 || {
    echo "Container runtime not found: ${CONTAINER_RUNTIME}" >&2
    exit 1
}

# ==================================================
# 4. Output and Cache Configuration
# ==================================================

mkdir -p \
    "${OUTPUT_DIR}" \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/vllm" \
    "${CACHE_ROOT}/torch"

export MODEL_PATH DATA_PATH ADAPTER_PATH_TEMPLATE OUTPUT_DIR
export ITERATION BASE_SEED VLLM_SEED FOLD

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCH_HOME="${CACHE_ROOT}/torch"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1

# ==================================================
# 5. Container Environment
# ==================================================

for name in \
    MODEL_PATH DATA_PATH ADAPTER_PATH_TEMPLATE OUTPUT_DIR \
    ITERATION BASE_SEED VLLM_SEED FOLD \
    HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE \
    VLLM_CACHE_ROOT TORCH_HOME \
    TOKENIZERS_PARALLELISM PYTHONUNBUFFERED \
    VLLM_NO_USAGE_STATS; do

    value="${!name}"
    export "SINGULARITYENV_${name}=${value}"
    export "APPTAINERENV_${name}=${value}"
done

export SINGULARITYENV_CONFIG_PATH="${CONFIG_PATH}"
export APPTAINERENV_CONFIG_PATH="${CONFIG_PATH}"

export SINGULARITYENV_CONTAINER_ENV_SCRIPT="${CONTAINER_ENV_SCRIPT}"
export APPTAINERENV_CONTAINER_ENV_SCRIPT="${CONTAINER_ENV_SCRIPT}"

# ==================================================
# 6. Bind Paths
# ==================================================

BIND_PATHS="${BASE_DIR}:${BASE_DIR},${DATA_DIR}:${DATA_DIR},${MODEL_PATH}:${MODEL_PATH},${OUTPUT_DIR}:${OUTPUT_DIR},${CACHE_ROOT}:${CACHE_ROOT},${ADAPTER_ROOT}:${ADAPTER_ROOT}"

CONTAINER_ARGS=(exec --nv --bind "${BIND_PATHS}")

if [[ -n "${OVERLAY_FILE}" ]]; then
    CONTAINER_ARGS+=(--overlay "${OVERLAY_FILE}:ro")
fi

# ==================================================
# 7. Job Information
# ==================================================

echo "=================================================="
echo "Fine-tuned Marshall Fire inference"
echo "Fold: ${FOLD}"
echo "Iteration: ${ITERATION}"
echo "Seed: ${VLLM_SEED}"
echo "Base model: ${MODEL_PATH}"
echo "Input CSV: ${DATA_PATH}"
echo "LoRA adapter: ${ADAPTER_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Configuration: ${CONFIG_PATH}"
echo "Container image: ${CONTAINER_IMAGE}"
echo "=================================================="

nvidia-smi -L || true

# ==================================================
# 8. Run Inference
# ==================================================

"${CONTAINER_RUNTIME}" "${CONTAINER_ARGS[@]}" "${CONTAINER_IMAGE}" \
    /bin/bash -c '
        set -euo pipefail

        if [[ -n "${CONTAINER_ENV_SCRIPT:-}" ]]; then
            if [[ ! -f "${CONTAINER_ENV_SCRIPT}" ]]; then
                echo "Environment script not found: ${CONTAINER_ENV_SCRIPT}" >&2
                exit 1
            fi

            source "${CONTAINER_ENV_SCRIPT}"
        fi

        export PYTHONNOUSERSITE=1

        cd "${1}"

        exec python3 -u stepwise_runner.py \
            --config "${2}" \
            --fold "${3}"
    ' bash "${BASE_DIR}" "${CONFIG_PATH}" "${FOLD}"

echo "Fold ${FOLD} completed: $(date)"
