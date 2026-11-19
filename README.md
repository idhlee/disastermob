# LLM Agents Fine-tuned with Mobility Data Improve Prediction of Disaster Evacuation Patterns

This repository contains the code and synthetic example datasets accompanying the research paper:

**LLM Agents Fine-tuned with Mobility Data Improve Prediction of Disaster Evacuation Patterns**

The repository provides implementations for:

- LoRA fine-tuning of large language models using mobility-informed synthetic agent data.
- Non-fine-tuned LLM agent simulations for disaster evacuation prediction.
- Fine-tuned LLM agent simulations using trained LoRA adapters.
- Synthetic example datasets for demonstrating the training and inference workflows.

## Repository Structure

```text
disastermob/
├── code/
│   ├── LoRA/
│   └── simulation/
│       ├── non-finetuned/
│       └── finetuned/
├── data/
│   └── synthetic/
│       ├── csv/
│       └── jsonl/
├── README.md
├── requirements.txt
├── environment.yml
└── LICENSE
```

## Installation

### Requirements

- Python 3.12
- Conda or Miniconda
- NVIDIA GPU for LLM training and inference
- Compatible CUDA environment and GPU drivers

### 1. Clone the repository

```bash
git clone https://github.com/idhlee/disastermob.git
cd disastermob
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate disastermob
```

### 4. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

The required Python packages and their pinned versions are specified in `requirements.txt`.

**Note:** GPU-based execution may require additional system-level CUDA, GPU driver, and container configuration depending on the computing environment.

## Getting Started

The repository provides three main workflows:

1. **LoRA Fine-tuning:** Train LLMs using mobility-informed synthetic agent data.
2. **Non-fine-tuned Simulation:** Run evacuation simulations using the original base model.
3. **Fine-tuned Simulation:** Run evacuation simulations using trained LoRA adapters.

Detailed instructions for each workflow are provided in the corresponding subdirectories.

## Data Availability

The repository includes synthetic example datasets to demonstrate the data structure and execution workflows.

The original mobility datasets used in this research, including SafeGraph and Cuebiq data, are subject to data access and sharing restrictions and are not redistributed in this repository.

The synthetic datasets are provided for demonstration and methodological inspection. They do not contain the original restricted mobility records.

See `data/synthetic/README.md` for further information.

## Reproducibility

This repository provides the implementation and configuration files needed to examine and execute the proposed training and simulation workflows.

The synthetic example datasets are intended to demonstrate the complete workflow, but they are not expected to reproduce the exact numerical results reported in the manuscript.

Reproducing the reported numerical results requires access to the corresponding underlying mobility datasets, the specified base models, and the relevant trained model checkpoints.

Details regarding model configurations, random seeds, checkpoint selection, and execution procedures are provided in the corresponding documentation.

The computational results obtained using the synthetic example datasets may differ from those reported in the manuscript.

## License

See [LICENSE](LICENSE) for license information.

## Citation

If you use this code, please cite the associated paper:

**LLM Agents Fine-tuned with Mobility Data Improve Prediction of Disaster Evacuation Patterns.**

Full bibliographic information will be provided upon publication.
