#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=47:59:59
#SBATCH --mem=128G
#SBATCH --job-name=nonft_marshall
#SBATCH --output=nonft_%j.out
#SBATCH --error=nonft_%j.err

set -euo pipefail

BASE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${BASE_DIR}/inference_config.yaml}"
RUNNER="${BASE_DIR}/stepwise_runner.py"

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-singularity}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to a Singularity/Apptainer image}"
OVERLAY_FILE="${OVERLAY_FILE:-}"
CONTAINER_ENV_SCRIPT="${CONTAINER_ENV_SCRIPT:-}"

MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/models/llama-3.1-8b-instruct}"
DATA_PATH="${DATA_PATH:-${BASE_DIR}/data/synthetic_data.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/outputs/non_finetuned}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_DIR}/cache}"

ITERATION="${ITERATION:-1}"
BASE_SEED="${BASE_SEED:-6000}"
VLLM_SEED="${VLLM_SEED:-$((BASE_SEED + ITERATION))}"

for path in \
    "${CONFIG_PATH}" \
    "${RUNNER}" \
    "${BASE_DIR}/model.py" \
    "${BASE_DIR}/agent.py" \
    "${MODEL_PATH}/config.json" \
    "${DATA_PATH}" \
    "${CONTAINER_IMAGE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path not found: ${path}" >&2
        exit 1
    fi
done

if [[ -n "${OVERLAY_FILE}" && ! -f "${OVERLAY_FILE}" ]]; then
    echo "Overlay file not found: ${OVERLAY_FILE}" >&2
    exit 1
fi

if ! command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1; then
    echo "Container runtime not found: ${CONTAINER_RUNTIME}" >&2
    exit 1
fi

mkdir -p \
    "${OUTPUT_DIR}" \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/vllm" \
    "${CACHE_ROOT}/torch" \
    "${BASE_DIR}/slurm_logs"

export MODEL_PATH DATA_PATH OUTPUT_DIR ITERATION BASE_SEED VLLM_SEED
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1

for name in \
    MODEL_PATH DATA_PATH OUTPUT_DIR ITERATION BASE_SEED VLLM_SEED \
    HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE VLLM_CACHE_ROOT \
    TORCH_HOME TOKENIZERS_PARALLELISM PYTHONUNBUFFERED VLLM_NO_USAGE_STATS; do
    value="${!name}"
    export "SINGULARITYENV_${name}=${value}"
    export "APPTAINERENV_${name}=${value}"
done

export SINGULARITYENV_CONFIG_PATH="${CONFIG_PATH}"
export APPTAINERENV_CONFIG_PATH="${CONFIG_PATH}"
export SINGULARITYENV_CONTAINER_ENV_SCRIPT="${CONTAINER_ENV_SCRIPT}"
export APPTAINERENV_CONTAINER_ENV_SCRIPT="${CONTAINER_ENV_SCRIPT}"

DATA_DIR="$(dirname "${DATA_PATH}")"
BIND_PATHS="${BASE_DIR}:${BASE_DIR},${DATA_DIR}:${DATA_DIR},${MODEL_PATH}:${MODEL_PATH},${OUTPUT_DIR}:${OUTPUT_DIR},${CACHE_ROOT}:${CACHE_ROOT}"

CONTAINER_ARGS=(exec --nv --bind "${BIND_PATHS}")

if [[ -n "${OVERLAY_FILE}" ]]; then
    CONTAINER_ARGS+=(--overlay "${OVERLAY_FILE}:ro")
fi

echo "=================================================="
echo "Non-fine-tuned Marshall Fire inference"
echo "Iteration: ${ITERATION}"
echo "Seed: ${VLLM_SEED}"
echo "Base model: ${MODEL_PATH}"
echo "Input CSV: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Configuration: ${CONFIG_PATH}"
echo "LoRA adapter: NONE"
echo "Container image: ${CONTAINER_IMAGE}"
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
        exec python3 -u stepwise_runner.py --config "${2}"
    ' bash "${BASE_DIR}" "${CONFIG_PATH}"

echo "Iteration ${ITERATION} completed: $(date)"
