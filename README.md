Code for the paper:
"Behavioral Fine-Tuning of LLM Agents Improves Predictability of Disaster Evacuation Decisions"

### Overview
This repository contains the code used to simulate evacuation behavior
using large language model (LLM) agents and mobility-based personas.

### Repository Structure
code/
├── LoRA/
│   ├── LoRA_training.py        # LoRA fine-tuning script
│   └── LoRA_training.slurm     # SLURM job script for HPC training
│
└── simulation/
    ├── finetuned/              # evacuation simulation using fine-tuned models
    └── non-finetuned/          # evacuation simulation using base models

LICENSE                         # repository license
README.md                       # project documentation
