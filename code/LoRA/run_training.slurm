
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:59
#SBATCH --mem=128G
#SBATCH --array=1-5%5
#SBATCH --job-name=llm_lora_5fold
#SBATCH --output=slurm_logs/training_%A_%a.out
#SBATCH --error=slurm_logs/training_%A_%a.err

set -euo pipefail

BASE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
TRAIN_SCRIPT="${BASE_DIR}/train_lora.py"
CONFIG_FILE="${TRAINING_CONFIG:-${BASE_DIR}/training_config.yaml}"

FOLD="${SLURM_ARRAY_TASK_ID:?Submit as Slurm array 1-5}"
[[ "${FOLD}" =~ ^[1-5]$ ]] || {
    echo "Invalid fold: ${FOLD}" >&2
    exit 1
}

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-singularity}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to your container image}"

DATA_DIR="${DATA_DIR:-${BASE_DIR}/../../data/synthetic/csv}"
JSONL_DIR="${JSONL_DIR:-${BASE_DIR}/../../data/synthetic/jsonl}"
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/models/llama-3.1-8b-instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/outputs/training}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_DIR}/cache}"

OVERLAY_FILE="${OVERLAY_FILE:-}"
CONTAINER_ENV_SCRIPT="${CONTAINER_ENV_SCRIPT:-}"

for path in \
    "${TRAIN_SCRIPT}" \
    "${CONFIG_FILE}" \
    "${DATA_DIR}/fold${FOLD}_train.csv" \
    "${DATA_DIR}/fold${FOLD}_test.csv" \
    "${JSONL_DIR}/fold${FOLD}_train.jsonl" \
    "${MODEL_PATH}/config.json" \
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

mkdir -p \
    "${OUTPUT_ROOT}" \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/huggingface/datasets" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/xdg"

export TRAINING_CONFIG="${CONFIG_FILE}"
export DATA_DIR JSONL_DIR MODEL_PATH OUTPUT_ROOT CACHE_ROOT

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

for name in \
    TRAINING_CONFIG DATA_DIR JSONL_DIR MODEL_PATH OUTPUT_ROOT CACHE_ROOT \
    HF_HOME HF_DATASETS_CACHE TORCH_HOME XDG_CACHE_HOME \
    TOKENIZERS_PARALLELISM PYTHONUNBUFFERED; do

    value="${!name}"
    export "SINGULARITYENV_${name}=${value}"
    export "APPTAINERENV_${name}=${value}"
done

export SINGULARITYENV_CONTAINER_ENV_SCRIPT="${CONTAINER_ENV_SCRIPT}"
export APPTAINERENV_CONTAINER_ENV_SCRIPT="${CONTAINER_ENV_SCRIPT}"

BIND_PATHS="${BASE_DIR}:${BASE_DIR},${DATA_DIR}:${DATA_DIR},${JSONL_DIR}:${JSONL_DIR},${MODEL_PATH}:${MODEL_PATH},${OUTPUT_ROOT}:${OUTPUT_ROOT},${CACHE_ROOT}:${CACHE_ROOT}"

CONTAINER_ARGS=(exec --nv --bind "${BIND_PATHS}")

if [[ -n "${OVERLAY_FILE}" ]]; then
    CONTAINER_ARGS+=(--overlay "${OVERLAY_FILE}:ro")
fi

echo "=================================================="
echo "LoRA training"
echo "Fold: ${FOLD}"
echo "Data: ${DATA_DIR}"
echo "JSONL: ${JSONL_DIR}/fold${FOLD}_train.jsonl"
echo "Base model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_ROOT}/fold${FOLD}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Overlay: ${OVERLAY_FILE:-NONE}"
echo "=================================================="

nvidia-smi -L || true

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

        exec python3 -u train_lora.py \
            --fold "${2}" \
            --model-path "${3}" \
            --output-root "${4}"
    ' bash "${BASE_DIR}" "${FOLD}" "${MODEL_PATH}" "${OUTPUT_ROOT}"

echo "Fold ${FOLD} completed: $(date)"
