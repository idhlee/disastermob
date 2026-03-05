# model.py
import pandas as pd
from mesa import Model
from mesa.time import RandomActivation
from agent import LLMAgent
import datetime
from vllm import LLM
from vllm.lora.request import LoRARequest
import os

os.makedirs("*/vllm_cache", exist_ok=True)
os.makedirs("*/hf_cache", exist_ok=True)

os.environ["VLLM_CACHE_ROOT"] = "*/vllm_cache"
os.environ["HF_HOME"] = "*/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "*/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "*/hf_cache"

BASE_MODEL_PATH = "*/llama-3.1-8b-instruct"

print("Loading vLLM one time (singleton)...")
print(f"Model path: {BASE_MODEL_PATH}")
print(f"vLLM cache: */vllm_cache")
print(f"HF cache: */hf_cache")

LLM_ENGINE = LLM(
    model=BASE_MODEL_PATH,
    dtype="bfloat16",
    trust_remote_code=True,
    enable_lora=True,
    gpu_memory_utilization=0.90,
)

# ==========================
# Mesa Model
# ==========================
def ordinal(n):
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    else:
        return f"{n}{['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"

class LLMModel(Model):
    def __init__(self, df, included_keys=[]):
        super().__init__()
        self.included_keys = included_keys
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.schedule = RandomActivation(self)
        self.evacuated_agents = []

        print(f" Total agents to process: {len(self.df)}")

        self.agent_list = []
        for i, agent_data in enumerate(self.df.to_dict(orient="records")):
            agent = LLMAgent(
                unique_id=i,
                model_mesa=self,
                agent_data=agent_data,
                included_keys=self.included_keys,
                llm_engine=LLM_ENGINE
            )
            self.schedule.add(agent)
            self.agent_list.append(agent)

        self.fire_start_date = datetime.date(2021, 12, 30)
        self.fire_current_date = self.fire_start_date

    def step(self):
        self.current_step += 1
        self.fire_current_date = self.fire_start_date + datetime.timedelta(weeks=self.current_step - 1)
        print(f"\n Model Step {self.current_step} Start")
        self.schedule.step()
        print(" Model Step End")
