# stepwise_runner.py
import os
import sys
import pandas as pd
import torch
import threading
import time
from model import LLMModel
import re

if len(sys.argv) > 1:
    only_run_exp = sys.argv[1]
    print(f" Only running stepwise experiment: {only_run_exp}")
else:
    only_run_exp = None

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cwd = os.getcwd()
match = re.search(r"Mark(\d+)", cwd)
if not match:
    mark_number = 29 
else:
    mark_number = int(match.group(1))

DATA_PATH = "*/df_caldor_final.csv" 
NUM_STEPS = 1

STEPWISE_EXPERIMENTS = {
    "E1": ["A1", "A2", "A3", "A4", "A5", "A6"],          # + population + income + white population + distance + fire direction + alerts  (Baseline)
    "E2": ["A1", "A3", "A4", "A5", "A6"],                # +            + income + white population + distance + fire direction + alerts
    "E3": ["A1", "A2", "A4", "A5", "A6"],                # + population +        + white population + distance + fire direction + alerts
    "E4": ["A1", "A2", "A3", "A5", "A6"],                # + population + income +                  + distance + fire direction + alerts
    "E5": ["A1", "A5", "A6"],                            # +            +        +                  + distance + fire direction + alerts
    "E6": ["A2", "A3", "A4", "A5", "A6"],                # + population + income + white population +          + fire direction + alerts
    "E7": ["A1", "A2", "A3", "A4", "A6"],                # + population + income + white population + distance +                + alerts      
    "E8": ["A1", "A2", "A3", "A4", "A5", "A6"],          # + population + income + white population + distance + fire direction + 
    "E9": ["A2", "A3", "A4"],                            # + population + income + white population +          +                +       
}

SKIP_EXPERIMENTS = set()

for exp_name, included_keys in STEPWISE_EXPERIMENTS.items():
    if exp_name in SKIP_EXPERIMENTS:
        print(f"Skipping already completed experiment: {exp_name}")
        continue
    if only_run_exp and exp_name != only_run_exp:
        continue

    print(f"\n🔬 Experiment: {exp_name} (Included variables: {included_keys})")
    run_output_dir = os.path.join(OUTPUT_DIR, f"{exp_name}")
    os.makedirs(run_output_dir, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    model = LLMModel(df=df, included_keys=included_keys)

    for step in range(1, NUM_STEPS + 1):
        print(f"\n Step {step} 시작\n") 
        model.step()

        evacuated_data, non_evacuated_data = [], []
        for agent in model.agent_list:
            record = {
                "agent_id": agent.uid, 
                "fire_distance": agent.home_fire_distance,
            }
            if agent.evacuation_status:
                record["evacuation_step"] = agent.evacuation_step
                evacuated_data.append(record)
            else:
                non_evacuated_data.append(record)

        suffix = f"_step_{step}.csv"
        pd.DataFrame(evacuated_data).to_csv(os.path.join(run_output_dir, f"evacuated_agents{suffix}"), index=False)
        pd.DataFrame(non_evacuated_data).to_csv(os.path.join(run_output_dir, f"non_evacuated_agents{suffix}"), index=False)

        print(f" Step {step} Done – {len(evacuated_data)} evacuated / {len(non_evacuated_data)} not evacuated")

    print(f"\n Experiment {exp_name} Summary:")
    print(f"Total evacuated: {len(evacuated_data)}") 
    print(f"Total non-evacuated: {len(non_evacuated_data)}")
