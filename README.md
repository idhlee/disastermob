Code for the paper:
"Behavioral Fine-Tuning of LLM Agents Improves Predictability of Disaster Evacuation Decisions"

### Overview
This repository contains the code used to simulate evacuation behavior using large language model (LLM) agents and mobility-based personas.
The repository includes scripts for LoRA fine-tuning of LLMs and agent-based evacuation simulations used in the experiments.

### Repository Structure

```text
code/
├── LoRA/
│   ├── LoRA_training.py
│   └── LoRA_training.slurm
│
└── simulation/
    ├── non-finetuned/
    │   ├── agent.py
    │   ├── agent_caldor.py
    │   ├── model.py
    │   ├── run_mesa.slurm
    │   └── stepwise_runner.py
    │
    └── finetuned/
        ├── agent.py
        ├── model.py
        ├── run_mesa.slurm
        └── stepwise_runner.py

LICENSE
README.md
```             

### Requirements

The code was developed using Python 3.10. The main dependencies include:

- mesa
- torch
- transformers
- peft
- accelerate
- vllm
- pandas
- numpy


### Fine-tuning LLMs

LoRA-based fine-tuning of the LLM agents is implemented in:

code/LoRA/LoRA_training.py

Training jobs were executed on an HPC cluster using the provided SLURM script:

code/LoRA/LoRA_training.slurm

### Running the Simulation

Evacuation simulations are implemented using the Mesa agent-based modeling framework.

Simulations using non-finetuned LLM agents can be executed from: code/simulation/non-finetuned/

Simulations using finetuned LLM agents can be executed from: code/simulation/finetuned/

SLURM job script: run_mesa.slurm

### Data

Mobility data used in this study are subject to data sharing restrictions and cannot be publicly released. Synthetic persona inputs used in the simulations were generated from aggregated mobility statistics as described in the paper.

### License

This project is released under the terms of the LICENSE file included in the repository.
