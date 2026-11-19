# Synthetic Example Data

This directory contains synthetic example datasets provided to demonstrate the data structure and execution workflows of the LLM-based disaster evacuation simulation framework.

All datasets are synthetic and are intended for methodological inspection, testing, and demonstration purposes.

## Directory Structure

```text
synthetic/
├── README.md
├── csv/
│   ├── synthetic_data.csv
│   ├── fold1_train.csv
│   ├── fold1_test.csv
│   ├── ...
│   ├── fold5_train.csv
│   └── fold5_test.csv
└── jsonl/
    ├── fold1_train.jsonl
    ├── fold1_test.jsonl
    ├── ...
    ├── fold5_train.jsonl
    └── fold5_test.jsonl
```

## 1. CSV Datasets

The `csv/` directory contains the synthetic agent-level dataset organized in two different formats for the non-fine-tuned and fine-tuned simulation workflows.

**`synthetic_data.csv`**

This file contains the complete synthetic dataset and is used for non-fine-tuned (base-model) LLM agent simulations.

**`fold{n}_train.csv` and `fold{n}_test.csv`**

These files are derived from the same underlying synthetic dataset as `synthetic_data.csv`, but are organized into five fold-specific training and test partitions.

They are used for the fine-tuned simulation workflow, including LoRA training and subsequent inference.

Each fold consists of:
- `fold{n}_train.csv`: Training partition used for LoRA fine-tuning.
- `fold{n}_test.csv`: Held-out test partition used for inference with the corresponding fine-tuned model.

The five-fold organization follows a spatially separated training and testing design at the census tract level.

Importantly, `synthetic_data.csv` and the fold-specific CSV files do not represent different independently generated datasets. They contain the same underlying synthetic data, organized differently according to the requirements of each workflow.

## 2. JSONL Datasets

The `jsonl/` directory contains prompt-formatted versions of the fold-specific datasets for LLM fine-tuning and inference.

Each JSONL record contains the following components:

| Field | Description |
|---|---|
| `System_message` | System instructions defining the disaster evacuation prediction task. |
| `Input1: Lifestyle` | Synthetic individual lifestyle and mobility information. |
| `Input2: Environment` | Neighborhood socioeconomic and fire-related contextual information. |
| `Input3: Disaster Information` | Disaster scenario description. |
| `Output` | Ground-truth evacuation decision included in training records. |

For each fold:
- `fold{n}_train.jsonl` contains the formatted training inputs and corresponding evacuation labels.
- `fold{n}_test.jsonl` contains the formatted test inputs without ground-truth output labels.

These JSONL files are used by the LoRA fine-tuning pipeline and its corresponding evaluation workflow.

## 3. Intended Usage

The datasets are organized according to the following workflows:

| Dataset | Intended workflow | Purpose |
|---|---|---|
| `synthetic_data.csv` | Non-fine-tuned simulation | Base-model inference using the complete synthetic dataset. |
| `fold{n}_train.csv` | Fine-tuned simulation | Input data for fold-specific LoRA training. |
| `fold{n}_test.csv` | Fine-tuned simulation | Input data for inference using the corresponding fold-specific adapter. |
| `fold{n}_train.jsonl` | LoRA fine-tuning | Prompt-formatted training examples with evacuation labels. |
| `fold{n}_test.jsonl` | Fine-tuned workflow | Prompt-formatted test examples without evacuation labels. |

## 4. Data Limitations and Reproducibility

All datasets provided in this repository are synthetic example data.

They are included to demonstrate the data structure, prompt construction, training procedures, and inference workflows implemented in this repository.

The synthetic datasets are provided as illustrative examples of the data structure and computational workflows. Results obtained using these datasets may differ from those reported in the associated manuscript, as the original analyses were conducted using the underlying research datasets.

Reproducing the reported numerical results requires access to the corresponding underlying mobility datasets and the relevant model configurations and trained checkpoints.

The original SafeGraph and Cuebiq mobility datasets are subject to data access and sharing restrictions and are not redistributed in this repository.

## 5. Usage Instructions

For detailed instructions on running each workflow, refer to:

- `code/simulation/non-finetuned/` for base-model simulations.
- `code/LoRA/` for LoRA fine-tuning.
- `code/simulation/finetuned/` for fine-tuned LLM agent simulations.
