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

See the installation instructions and environment specifications in `requirements.txt` and `environment.yml`.

## Getting Started

The repository provides three main workflows:

1. **LoRA Fine-tuning:** Train LLMs using mobility-informed synthetic agent data.
2. **Non-fine-tuned Simulation:** Run evacuation simulations using the original base model.
3. **Fine-tuned Simulation:** Run evacuation simulations using trained LoRA adapters.

Detailed instructions for each workflow are provided in the corresponding subdirectories.

## Data Availability

The repository includes synthetic example datasets to demonstrate the data structure and execution workflows.

The original mobility datasets used in this research, including SafeGraph and Cuebiq data, are subject to data access and sharing restrictions and are not redistributed in this repository.

See `data/synthetic/README.md` for further information.

## Reproducibility

The repository is intended to facilitate methodological inspection and execution of the provided workflows.

The synthetic example datasets are provided for demonstration purposes and are not expected to reproduce the exact numerical results reported in the manuscript. Reproducing the reported results requires access to the corresponding underlying mobility datasets and the specified model checkpoints.

Further details regarding model configurations, random seeds, checkpoint selection, and execution procedures are provided in the relevant documentation.

## License

See [LICENSE](LICENSE) for license information.

## Citation

If you use this code, please cite the associated paper:

**LLM Agents Fine-tuned with Mobility Data Improve Prediction of Disaster Evacuation Patterns.**

Full bibliographic information will be provided upon publication.
