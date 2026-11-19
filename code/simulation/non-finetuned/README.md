# Non-fine-tuned LLM Simulation

This directory contains the implementation of non-fine-tuned LLM simulations for predicting individual evacuation decisions during the 2021 Marshall Fire.

The simulation framework uses Meta-Llama-3.1-8B-Instruct without additional fine-tuning. It integrates Mesa for simulation and vLLM for LLM inference.

This workflow serves as the base-model comparison for the fine-tuned LLM simulation framework.

## 1. Overview

The non-fine-tuned simulation workflow consists of the following steps:

1. Load the pretrained Llama-3.1-8B-Instruct model without LoRA adapters.
2. Load the complete synthetic agent dataset.
3. Construct individual prompts using lifestyle, neighborhood, and disaster-related information.
4. Generate evacuation decisions using the base LLM.
5. Extract the evacuation decisions from the generated responses.
6. Save individual predictions and experiment summaries.

Unlike the fine-tuned simulation workflow, this implementation does not require LoRA training or fold-specific model adapters.

## 2. Directory Structure

```text
non-finetuned/
├── README.md
├── agent.py
├── model.py
├── stepwise_runner.py
├── inference_config.yaml
└── run_non_finetuned.slurm
```

The files serve the following purposes:

| File | Description |
|---|---|
| `agent.py` | Defines LLM agents, constructs prompts, generates responses, and extracts evacuation decisions. |
| `model.py` | Implements the Mesa-based simulation and agent scheduling. |
| `stepwise_runner.py` | Loads the synthetic dataset, executes experiments, and saves predictions. |
| `inference_config.yaml` | Defines model settings, generation parameters, prompts, and experiment configurations. |
| `run_non_finetuned.slurm` | Executes the non-fine-tuned simulation using Slurm. |

## 3. Base Model

The simulation uses the following model configuration:

- **Model:** Meta-Llama-3.1-8B-Instruct
- **Inference method:** Base-model inference without fine-tuning
- **Inference engine:** vLLM
- **Simulation framework:** Mesa
- **Precision:** BF16

The base model must be obtained separately and made available locally before running inference.

The model is loaded directly from its local directory without applying any LoRA adapters or other fine-tuned weights.

The base model should be consistent with the model used in the fine-tuned simulation workflow to enable a meaningful comparison.

## 4. Input Data

The simulation uses the complete synthetic dataset provided in:

```text
data/synthetic/csv/
└── synthetic_data.csv
```

Unlike the fine-tuned workflow, which evaluates fold-specific held-out test datasets, the non-fine-tuned simulation uses the complete synthetic dataset.

Each row corresponds to a synthetic individual with contextual information used to construct the evacuation prediction prompt.

The required columns are:

| Column | Description |
|---|---|
| `persona_id` | Unique identifier for each synthetic individual. |
| `geoid` | Geographic identifier of the residential area. |
| `tract_id` | Census tract identifier. |
| `persona_output` | Natural-language description of lifestyle and mobility patterns. |
| `distance_to_fire_km` | Distance from the residential location to the fire ignition location. |
| `direction_from_fire` | Direction relative to the fire. |
| `total_population_category` | Neighborhood population category. |
| `median_income_category` | Neighborhood income category. |
| `white_population_category` | Neighborhood racial composition category. |
| `alert_level` | Initial evacuation alert information. |

The input dataset is used to construct the individual simulation prompts.

The non-fine-tuned inference workflow does not require the JSONL training files or the fold-specific LoRA adapters.

## 5. Simulation Framework

The simulation framework is implemented using Mesa and integrates the pretrained LLM to predict evacuation decisions during the Marshall Fire.

Each individual is represented by an `LLMAgent`, which incorporates lifestyle, mobility, neighborhood, and disaster-related attributes.

For each individual, the simulation follows these steps:

1. Construct a prompt using the individual's contextual information.
2. Format the prompt using the model tokenizer's native chat template.
3. Generate a response using the pretrained LLM.
4. Extract the evacuation decision from the generated response.
5. Record the predicted decision and complete generated response.

The simulation is configured to evaluate the initial evacuation decision using a single simulation step.

The primary evacuation decisions are:

- `evacuate`
- `not evacuate`

The generated responses and corresponding evacuation predictions are saved for subsequent analysis.

## 6. Prompt Construction

Each individual receives a prompt consisting of a system message and a user message.

The user message contains three major components:

**Lifestyle information**

A natural-language description of the individual's lifestyle and mobility patterns.

**Neighborhood and spatial information**

Selected socioeconomic characteristics, distance from the fire, relative direction, and initial evacuation alert information.

**Disaster information**

A description of the 2021 Marshall Fire and the disaster conditions available in the simulation scenario.

The model is instructed to explain its reasoning before providing a final evacuation decision in one of the following formats:

- `Final answer: Evacuate`
- `Final answer: Not evacuate`

The generated response and extracted evacuation decision are recorded in the simulation outputs.

The prompt structure is consistent with the fine-tuned simulation workflow, allowing comparison between the base model and its fine-tuned counterpart.

## 7. Feature Ablation Experiments

The simulation supports nine predefined feature configurations specified in `inference_config.yaml`.

The six contextual features are:

| Feature | Description |
|---|---|
| A1 | Distance from the fire |
| A2 | Neighborhood total population category |
| A3 | Neighborhood median household income category |
| A4 | Neighborhood white population composition |
| A5 | Direction relative to the fire |
| A6 | Initial evacuation alert |

The predefined experiments are:

| Experiment | Included Features |
|---|---|
| E1 | A1, A2, A3, A4, A5, A6 |
| E2 | A1, A3, A4, A5, A6 |
| E3 | A1, A2, A4, A5, A6 |
| E4 | A1, A2, A3, A5, A6 |
| E5 | A1, A5, A6 |
| E6 | A2, A3, A4, A5, A6 |
| E7 | A1, A2, A3, A4, A6 |
| E8 | A1, A2, A3, A4, A5 |
| E9 | A2, A3, A4 |

E1 represents the complete feature configuration.

E2–E9 exclude individual contextual features or combinations of features to examine their contributions to predicted evacuation behavior.

By default, the simulation executes all nine experiments using the same synthetic dataset.

## 8. Inference Configuration

The inference settings are specified in `inference_config.yaml`.

### Model Configuration

| Parameter | Value |
|---|---|
| Base model | Meta-Llama-3.1-8B-Instruct |
| Precision | BF16 |
| Maximum model length | 6,144 |
| GPU memory utilization | 0.90 |
| Maximum concurrent sequences | 64 |
| LoRA adapters | Not used |

### Generation Configuration

| Parameter | Value |
|---|---|
| Maximum generated tokens | 1,024 |
| Temperature | 0.8 |
| Top-p | 0.85 |
| Top-k | 30 |
| Number of outputs per request | 1 |

These settings are used by vLLM to generate evacuation predictions for each individual.

### Simulation Configuration

| Parameter | Value |
|---|---|
| Iteration | 1 |
| Base seed | 6000 |
| Inference seed | 6001 |
| Simulation steps | 1 |
| Maximum agents | No limit by default |

The inference seed is calculated as:

```python
seed = base_seed + iteration
```

The provided Slurm script executes one iteration by default.

## 9. Configuration and Paths

The simulation uses `inference_config.yaml` to define the model, dataset, and output directories.

The configuration includes:

```yaml
paths:
  model_path: "./models/llama-3.1-8b-instruct"
  data_csv: "./data/synthetic_data.csv"
  output_dir: "./outputs/non_finetuned"
```

These paths can be replaced with locations appropriate for the user's computing environment.

For direct execution, relative configuration paths are interpreted relative to the location of `inference_config.yaml`.

### Environment Variables

The following environment variables can be used to override the configured paths:

| Variable | Description |
|---|---|
| `MODEL_PATH` | Local directory containing the base model. |
| `DATA_PATH` | Path to the complete synthetic CSV dataset. |
| `OUTPUT_DIR` | Directory for simulation outputs. |
| `CONFIG_PATH` | Configuration file path used by the Slurm script. |
| `CONTAINER_IMAGE` | Container image for execution. |
| `CONTAINER_RUNTIME` | Container runtime, defaulting to Singularity. |
| `OVERLAY_FILE` | Optional container overlay. |
| `CONTAINER_ENV_SCRIPT` | Optional environment initialization script. |
| `CACHE_ROOT` | Cache directory for inference dependencies. |

The model, dataset, output, and container paths can be configured without modifying the Python scripts.

## 10. Execution

### 10.1 Environment Setup

Refer to the repository-level `README.md`, `requirements.txt`, and `environment.yml` for installation instructions.

The simulation requires a compatible CUDA-enabled GPU environment, access to the pretrained Llama-3.1-8B-Instruct model, and the complete synthetic dataset.

For HPC execution, a compatible container runtime and container image are also required.

### 10.2 Configure Paths

From the repository root, navigate to the non-fine-tuned simulation directory:

```bash
cd code/simulation/non-finetuned
```

Configure the model, dataset, output, and container paths:

```bash
export MODEL_PATH="/path/to/llama-3.1-8b-instruct"
export DATA_PATH="$(realpath ../../../data/synthetic/csv/synthetic_data.csv)"
export OUTPUT_DIR="$(pwd)/outputs/non_finetuned"
export CONTAINER_IMAGE="/path/to/container.sif"
```

Replace the example model and container paths with the appropriate locations in your computing environment.

The following optional variables can also be configured when needed:

```bash
export CONTAINER_RUNTIME="singularity"
export OVERLAY_FILE="/path/to/overlay.ext3"
export CONTAINER_ENV_SCRIPT="/path/to/env.sh"
```

The optional overlay and environment initialization script depend on the target HPC environment.

### 10.3 Full Non-fine-tuned Simulation

The non-fine-tuned simulation processes the complete synthetic dataset using the pretrained base model.

Unlike the fine-tuned workflow, this execution does not require fold-specific inference or a five-fold Slurm job array.

Submit the simulation job:

```bash
mkdir -p slurm_logs
sbatch run_non_finetuned.slurm
```

The Slurm script runs one simulation job using one GPU.

By default, the simulation executes one iteration of all nine configured feature experiments.

### 10.4 Simulation Completion

Job execution can be monitored using:

```bash
squeue -u "$USER"
```

The Slurm output and error logs are saved under `slurm_logs/`.

After successful completion, the prediction files and experiment summaries are available under the configured output directory.

## 11. Output Structure

Simulation outputs are organized by iteration and experiment:

```text
outputs/
└── non_finetuned/
    └── iteration_1/
        ├── E1/
        │   ├── agent_predictions.csv
        │   └── summary.csv
        ├── E2/
        ├── ...
        └── E9/
```

### Individual Predictions

Each `agent_predictions.csv` contains the individual simulation results.

The output fields include:

| Column | Description |
|---|---|
| `iteration` | Simulation iteration. |
| `experiment` | Feature configuration identifier. |
| `agent_id` | Synthetic individual identifier. |
| `geoid` | Geographic identifier. |
| `tract_id` | Census tract identifier. |
| `fire_distance_km` | Distance to the fire. |
| `direction_from_fire` | Direction relative to the fire. |
| `alert_level` | Initial evacuation alert information. |
| `predicted_decision` | Extracted evacuation decision. |
| `predicted_evacuated` | Evacuation prediction. |
| `ground_truth_evacuated` | Optional reference outcome. |
| `ground_truth_label` | Optional reference label. |
| `inference_seconds` | LLM inference duration. |
| `generated_response` | Complete generated LLM response, including any reasoning text. |

### Experiment Summary

Each `summary.csv` contains experiment-level statistics, including the number of individuals, predicted evacuations, predicted non-evacuations, predicted evacuation rate, and inference seed.

The predicted evacuation rate is calculated as the proportion of predicted evacuations among the successfully classified evacuation decisions.

## 12. Reproducibility Notes

The provided implementation documents the base-model inference settings, prompt construction, feature ablation configurations, and simulation output formats.

The synthetic dataset included in this repository is intended to demonstrate the computational workflow.

Results obtained using the synthetic dataset may differ from those reported in the associated manuscript because the original experiments were conducted using the underlying research datasets.

Reproducing the original numerical results requires access to the relevant research datasets, the same base-model checkpoint, and a compatible computational environment.
