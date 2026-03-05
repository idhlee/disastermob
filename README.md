Code for the paper:
"Behavioral Fine-Tuning of LLM Agents Improves Predictability of Disaster Evacuation Decisions"

### Overview
This repository contains the code used to simulate evacuation behavior
using large language model (LLM) agents and mobility-based personas.

### Repository Structure
code/
├── LoRA/
│   ├── LoRA_training.py        # LoRA fine-tuning of LLM
│   └── LoRA_training.slurm     
│
└── simulation/
    ├── non-finetuned/          # evacuation simulation using pre-trained LLM (Marshall Fire)
    │   ├── agent.py
    │   ├── agent_caldor.py
    │   ├── model.py
    │   ├── run_mesa.slurm
    │   └── stepwise_runner.py
    │
    └── finetuned/              # evacuation simulation using fine-tuned LLM (Caldor Fire)
        ├── agent.py
        ├── model.py
        ├── run_mesa.slurm
        └── stepwise_runner.py

LICENSE                         # repository license
README.md                       # project documentation
