# LoRA Fine-tuning

This directory contains the implementation of Low-Rank Adaptation (LoRA) fine-tuning for LLM-based disaster evacuation prediction.

The training pipeline uses mobility-informed synthetic agent data and a five-fold spatial holdout design to fine-tune LLMs for predicting evacuation decisions.

## 1. Base Model

The training script is configured for:

- **Model:** Meta-Llama-3.1-8B-Instruct
- **Fine-tuning method:** LoRA
- **Task:** Causal language modeling
- **Precision:** BF16
- **Hardware:** NVIDIA H200 GPU

The base model must be available locally before training. Its location is specified through `MODEL_PATH` or `training_config.yaml`.

The script loads the model from a local directory rather than downloading a model directly during training.

## 2. Directory Structure

```text
LoRA/
├── README.md
├── train_lora.py
├── run_training.slurm
└── training_config.yaml
```

The training pipeline uses the following datasets:

```text
data/
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

The CSV and JSONL directories are configured independently through environment variables.

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

- `System_message`
- `Input1: Lifestyle`
- `Input2: Environment`
- `Input3: Disaster Information`
- `Output`

The `Output` field must contain one of the following labels:

- `Final answer: Evacuate`
- `Final answer: Not evacuate`

**Data alignment**

The CSV and JSONL training files must contain exactly the same records in the same row order. The training script checks the record counts and validates the alignment of the first 32 records by default.

The test CSV is used to verify geographic separation from the training data. Held-out test outcomes are not used for model fitting or checkpoint selection.

## 4. Five-Fold Spatial Holdout

The training pipeline uses five predefined outer folds, with geographic separation at the census tract level.

For each fold:

1. The outer training and test partitions are loaded separately.
2. The training partition is further divided into LoRA fitting and internal validation subsets.
3. The internal split is performed at the census tract level using `GroupShuffleSplit`.
4. Approximately 80% of the outer-training tracts are used for LoRA fitting, while 20% are reserved for internal checkpoint validation.
5. The outer test partition remains excluded from training and checkpoint selection.

The internal validation data are used exclusively for monitoring training and selecting the best checkpoint.

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

The LoRA adapters are applied to the following modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

### Training objective

The training pipeline uses full-sequence cross-entropy loss. All non-padding tokens in the formatted conversation, including system instructions, user inputs, and the assistant response, contribute to the training loss.

The assistant's supervised response consists of the evacuation decision only:

- `Final answer: Evacuate`
- `Final answer: Not evacuate`

Reasoning explanations are not included in the training targets. The experiment therefore evaluates whether the model can generate explanations at inference time while being fine-tuned on decision-only responses.

The model's native chat template is used to format the training conversations.

## 6. Random Seeds and Checkpoint Selection

A base random seed of 42 is used.

The seed for each fold is defined as:

```python
seed = 42 + fold
```

Thus, folds 1–5 use seeds 43–47, respectively.

The same fold-specific seed is used for model training and data-related randomization.

Checkpoint selection is based on the lowest internal validation loss (`eval_loss`). The training pipeline enables `load_best_model_at_end`, and the selected model is saved as `final_adapter/` for each fold.

The outer test partition is not used for checkpoint selection.

## 7. Configuration and Paths

The training pipeline uses `training_config.yaml` to define portable default paths.

```yaml
paths:
  data_dir: "./data"
  jsonl_dir: "./data/jsonl"
  model_path: "./models/llama-3.1-8b-instruct"
  output_root: "./outputs/training"
```

These defaults can be overridden through environment variables:

| Variable | Description |
|---|---|
| `DATA_DIR` | Directory containing fold-specific CSV files. |
| `JSONL_DIR` | Directory containing fold-specific JSONL files. |
| `MODEL_PATH` | Local base-model directory. |
| `OUTPUT_ROOT` | Directory for training outputs. |
| `TRAINING_CONFIG` | Path to the YAML configuration file. |
| `CACHE_ROOT` | Root directory for model and dataset caches. |

This allows the pipeline to run in different computing environments without modifying the training script.

## 8. Execution

### 8.1 Environment Setup

Refer to the repository-level `README.md`, `requirements.txt`, and `environment.yml` for installation instructions and Python dependencies.

The training workflow requires a compatible CUDA environment, an NVIDIA H200 GPU, and access to the base model.

### 8.2 Data Validation

Before training, validate the data for a specific fold:

```bash
python train_lora.py --fold 1 --check-data-only
```

### 8.3 Preflight Check

To check tokenizer formatting, data alignment, padding, and training argument compatibility:

```bash
python train_lora.py --fold 1 --preflight-only
```

This performs a limited preflight check without loading the full training dataset or using a GPU.

### 8.4 Smoke Test

To execute a small end-to-end training test:

```bash
python train_lora.py --fold 1 --test-mode
```

The smoke test uses a limited data subset and a shortened training schedule.

### 8.5 Full Training

To train a specific fold directly:

```bash
python train_lora.py --fold 1
```

To execute all five folds using the provided Slurm job array:

```bash
mkdir -p slurm_logs
sbatch run_training.slurm
```

The Slurm script submits five fold-specific jobs using one GPU per job.

The container image and relevant paths must be configured through the corresponding environment variables before submission.

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
- `loss_history.csv`: Training and validation loss history.
- `tract_split_manifest.csv`: Geographic assignments for internal training, validation, and outer testing.
- `run_config.json`: Fold-specific training configuration and data split information.
- `metrics.json`: Training metrics, validation metrics, and selected checkpoint information.

The resulting adapters are used by the fine-tuned simulation pipeline in `code/simulation/finetuned/`.

## 10. Reproducibility Notes

The provided implementation documents the training configuration, fold-specific random seeds, geographic data splitting, and internal validation-based checkpoint selection.

Reproducing the original experimental results also requires the corresponding research datasets, the same base-model checkpoint, and compatible computational environments.

The synthetic example datasets included in this repository are intended to demonstrate the training workflow and are not substitutes for the original research datasets.
