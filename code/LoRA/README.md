# LoRA Fine-tuning

This directory contains the implementation of Low-Rank Adaptation (LoRA) fine-tuning for LLM-based disaster evacuation prediction.

The training pipeline uses mobility-informed synthetic agent data and a five-fold spatial holdout design to fine-tune LLMs for predicting evacuation decisions.

## 1. Base Model

The training script is configured for:

- **Model:** Meta-Llama-3.1-8B-Instruct
- **Fine-tuning method:** LoRA
- **Task:** Causal language modeling
- **Precision:** BF16
- **GPU requirement:** NVIDIA H200

The base model is not included in this repository and must be obtained separately before training.

The model is loaded from a local directory specified through `MODEL_PATH` or `training_config.yaml`. The training script does not automatically download the model or pin a remote Hugging Face revision. Users should use the intended model checkpoint consistently across experiments.

## 2. Directory Structure

```text
LoRA/
├── README.md
├── train_lora.py
├── run_training.slurm
└── training_config.yaml
```

The training pipeline uses the following datasets from the repository:

```text
disastermob/
├── code/
│   └── LoRA/
│       ├── README.md
│       ├── train_lora.py
│       ├── run_training.slurm
│       └── training_config.yaml
└── data/
    └── synthetic/
        ├── csv/
        │   ├── fold1_train.csv
        │   ├── fold1_test.csv
        │   └── ...
        └── jsonl/
            ├── fold1_train.jsonl
            ├── fold2_train.jsonl
            └── ...
```

## 3. Data Requirements

For each fold, the training pipeline requires:

**CSV files**

- `fold{n}_train.csv`
- `fold{n}_test.csv`

**JSONL files**

- `fold{n}_train.jsonl`

The training CSV must contain the following essential columns:

- `tract_id`
- `persona_output`
- `evacuation_label`
- `distance_to_fire_km`
- `direction_from_fire`

Each training JSONL record must contain:

| Field | Description |
|---|---|
| `System_message` | System instructions defining the disaster evacuation prediction task. |
| `Input1: Lifestyle` | Synthetic individual lifestyle and mobility information. |
| `Input2: Environment` | Neighborhood socioeconomic and fire-related contextual information. |
| `Input3: Disaster Information` | Disaster scenario description. |
| `Output` | Ground-truth evacuation decision used for supervised training. |

The `Output` field must contain one of the following labels:

- `Final answer: Evacuate`
- `Final answer: Not evacuate`

### Data Alignment

The CSV and JSONL training files must contain exactly the same records in the same row order.

The training script checks the record counts and validates the alignment of the first 32 records by default. The validation checks include lifestyle descriptions, evacuation labels, fire distance, and fire direction.

The outer test CSV is also checked for geographic separation from the training data. Its tract identifiers are used to verify that training and test tracts do not overlap.

Held-out test outcomes are not used for model fitting or checkpoint selection.

## 4. Five-Fold Spatial Holdout

The training pipeline uses five predefined outer folds, with geographic separation at the census tract level.

For each fold:

1. The outer training and test partitions are loaded separately.
2. The outer training partition is further divided into LoRA fitting and internal checkpoint-validation subsets.
3. The internal split is performed at the census tract level using `GroupShuffleSplit`.
4. Approximately 80% of the outer-training tracts are used for LoRA fitting, while 20% are reserved for internal validation.
5. The outer test partition remains excluded from model training and checkpoint selection.

The internal validation data are used exclusively for monitoring training performance and selecting the best checkpoint.

The random state for the internal split is determined by the fold-specific seed.

## 5. LoRA Training Configuration

The main training configuration is as follows:

| Parameter | Value |
|---|---|
| Base model | Llama-3.1-8B-Instruct |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| Learning rate | 2e-4 |
| Training epochs | 3 |
| Training batch size | 2 |
| Evaluation batch size | 2 |
| Gradient accumulation steps | 8 |
| Warmup steps | 10 |
| Maximum sequence length | 2,048 |
| Optimizer | AdamW |
| Precision | BF16 |
| Gradient checkpointing | Enabled |
| Logging interval | 10 steps |
| Evaluation interval | 300 steps |
| Checkpoint saving interval | 300 steps |
| Maximum saved checkpoints | 2 |

The LoRA adapters are applied to the following modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

### Training Objective

The training pipeline uses full-sequence cross-entropy loss.

All non-padding tokens in the formatted conversation, including system instructions, user inputs, and the assistant response, contribute to the training loss. Padding tokens are excluded from the loss calculation.

The assistant's supervised response consists of the evacuation decision only:

- `Final answer: Evacuate`
- `Final answer: Not evacuate`

Reasoning explanations are not included in the training targets. The experiment therefore examines whether the model can generate explanations at inference time while being fine-tuned on decision-only responses.

The model's native chat template is used to format the training conversations through `tokenizer.apply_chat_template()`.

## 6. Random Seeds and Checkpoint Selection

A base random seed of 42 is used.

The seed for each fold is defined as:

```python
seed = 42 + fold
```

Thus, folds 1–5 use seeds 43–47, respectively.

The fold-specific seed is applied to model training and data-related randomization. The internal geographic split also uses the corresponding fold-specific random state.

### Checkpoint Selection

Checkpoint selection is based on the lowest internal validation loss (`eval_loss`).

The training pipeline uses:

- `load_best_model_at_end=True`
- `metric_for_best_model="eval_loss"`
- `greater_is_better=False`

Evaluation and checkpoint saving are performed at the same step intervals.

After training, the selected best model is saved as `final_adapter/` for each fold.

The outer test partition is not used for checkpoint selection.

## 7. Configuration and Paths

The training pipeline uses `training_config.yaml` to define portable default paths.

All relative paths are resolved with respect to the location of `training_config.yaml`, not the current working directory.

The default configuration is:

```yaml
# Portable defaults. Paths are relative to this configuration file.
# Override them with DATA_DIR, JSONL_DIR, MODEL_PATH, and OUTPUT_ROOT as needed.

paths:
  data_dir: "../../data/synthetic/csv"
  jsonl_dir: "../../data/synthetic/jsonl"
  model_path: "../../models/llama-3.1-8b-instruct"
  output_root: "./outputs/training"
```

These defaults correspond to the repository directory structure.

If the repository structure is maintained and the base model is placed in the specified location, the default paths can be used without modification.

### Environment Variables

The following environment variables can be used to override the default paths:

| Variable | Description |
|---|---|
| `DATA_DIR` | Directory containing fold-specific CSV files. |
| `JSONL_DIR` | Directory containing fold-specific JSONL files. |
| `MODEL_PATH` | Local base-model directory. |
| `OUTPUT_ROOT` | Directory for saving training outputs. |
| `TRAINING_CONFIG` | Path to the YAML configuration file. |
| `CACHE_ROOT` | Root directory for model and dataset caches. |

For example:

```bash
export DATA_DIR="/path/to/data/synthetic/csv"
export JSONL_DIR="/path/to/data/synthetic/jsonl"
export MODEL_PATH="/path/to/llama-3.1-8b-instruct"
export OUTPUT_ROOT="/path/to/training/outputs"
```

Replace the example paths with the corresponding locations in your computing environment.

Users do not need to modify the Python training script to change these paths.

### Configuration for HPC Execution

For Slurm-based execution, the provided `run_training.slurm` script uses environment variables to configure the data, model, output, and cache directories.

The container image must be specified through `CONTAINER_IMAGE`.

The following additional variables are available:

| Variable | Description |
|---|---|
| `CONTAINER_RUNTIME` | Container runtime (default: `singularity`). |
| `CONTAINER_IMAGE` | Path to the container image. |
| `OVERLAY_FILE` | Optional container overlay file. |
| `CONTAINER_ENV_SCRIPT` | Optional container environment initialization script. |

The container image, runtime, GPU resources, and system-specific settings should be configured according to the target HPC environment.

## 8. Execution

### 8.1 Environment Setup

Refer to the repository-level `README.md`, `requirements.txt`, and `environment.yml` for installation instructions and Python dependencies.

The full training workflow requires:

- A compatible CUDA environment.
- An NVIDIA GPU with BF16 support (NVIDIA H200 recommended).
- Access to the Meta-Llama-3.1-8B-Instruct base model.
- The synthetic CSV and JSONL datasets provided in this repository.
- A compatible container runtime and container image for Slurm-based execution.

### 8.2 Configure Paths

Before submitting the training jobs, navigate to the LoRA directory:

```bash
cd code/LoRA
```

Configure the dataset, model, output, and container paths:

```bash
export DATA_DIR="$(realpath ../../data/synthetic/csv)"
export JSONL_DIR="$(realpath ../../data/synthetic/jsonl)"
export MODEL_PATH="/path/to/llama-3.1-8b-instruct"
export OUTPUT_ROOT="$(pwd)/outputs/training"
export CONTAINER_IMAGE="/path/to/container.sif"
```

Replace the example model and container paths with the appropriate locations in your computing environment.

The following optional variables can also be configured when needed:

```bash
export CONTAINER_RUNTIME="singularity"
export OVERLAY_FILE="/path/to/overlay.ext3"
export CONTAINER_ENV_SCRIPT="/path/to/env.sh"
```

The default container runtime is Singularity. Overlay and environment initialization scripts are optional and depend on the target HPC environment.

The dataset and model paths are passed to the training jobs through environment variables, without requiring modifications to the Python training script.

### 8.3 Full Five-Fold Training

The complete training experiment requires independently fine-tuning all five predefined spatial folds.

The provided `run_training.slurm` script uses a Slurm job array to execute all five folds:

```bash
mkdir -p slurm_logs
sbatch run_training.slurm
```

The job array is configured as follows:

```bash
#SBATCH --array=1-5%5
```

Each task is assigned to one fold:

| Slurm Task | Training Fold |
|---|---|
| 1 | Fold 1 |
| 2 | Fold 2 |
| 3 | Fold 3 |
| 4 | Fold 4 |
| 5 | Fold 5 |

Each fold is trained independently using one GPU. Up to five fold-specific jobs can run simultaneously, subject to available cluster resources.

After all five jobs complete successfully, five fold-specific LoRA adapters will be available for the subsequent fine-tuned simulation workflow.

### 8.4 Training Completion

Job execution can be monitored using:

```bash
squeue -u "$USER"
```

The Slurm output and error logs are saved in the `slurm_logs/` directory.

The training outputs are organized by fold under the configured output directory. Each fold should contain its corresponding `final_adapter/` directory after successful completion.

All five fold-specific adapters are required for the complete five-fold fine-tuned simulation workflow.

## 9. Output Structure

Training outputs are organized by fold:

```text
outputs/
└── training/
    ├── fold1/
    │   ├── checkpoints/
    │   ├── final_adapter/
    │   ├── training_log.txt
    │   ├── loss_history.csv
    │   ├── tract_split_manifest.csv
    │   ├── run_config.json
    │   └── metrics.json
    ├── fold2/
    ├── fold3/
    ├── fold4/
    └── fold5/
```

Important outputs include:

- `final_adapter/`: Selected LoRA adapter and tokenizer files.
- `checkpoints/`: Intermediate training checkpoints.
- `training_log.txt`: Training execution log.
- `loss_history.csv`: Training and internal validation loss history.
- `tract_split_manifest.csv`: Geographic assignments for internal training, validation, and outer testing.
- `run_config.json`: Fold-specific training configuration and data split information.
- `metrics.json`: Training metrics, internal validation metrics, and selected checkpoint information.

The resulting adapters are used by the fine-tuned simulation pipeline in `code/simulation/finetuned/`.

## 10. Reproducibility Notes

The provided implementation documents the training configuration, fold-specific random seeds, geographic data splitting, and internal validation-based checkpoint selection.

The training pipeline is designed to demonstrate the LoRA fine-tuning workflow using the synthetic example datasets provided in this repository.

Results obtained using these synthetic datasets may differ from those reported in the associated manuscript, as the original analyses were conducted using the underlying research datasets.

Reproducing the reported numerical results requires access to the corresponding research datasets, the same base-model checkpoint, and compatible computational environments.
