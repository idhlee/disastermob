# agent.py
import torch
import time
import re
import os
import pandas as pd
from mesa import Agent
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# ==========================================
#  캐시 디렉토리 설정 - scratch 고정 경로
# ==========================================
os.environ["VLLM_CACHE_ROOT"] = "/scratch/dl5683/vllm_cache"
os.environ["HF_HOME"] = "/scratch/dl5683/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/dl5683/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/dl5683/hf_cache"

torch.cuda.empty_cache()

# ==========================================
#  모델 설정
# ==========================================
BASE_MODEL_PATH = "/scratch/dl5683/llm/llama-3.1-8b-instruct"
LORA_MODEL_PATH = "/scratch/dl5683/pytorch-example/LoRA/lora_models/30_percent_newnew"

sampling_params = SamplingParams(
    max_tokens=1024,
    temperature=0.8,
    top_p=0.85,
    top_k=30,
    n=1
)

# ==========================================
#  SYSTEM MESSAGE + INSTRUCTIONS
# ==========================================
system_message = (
    "You are a disaster behavior analyst. Your job is to assess whether an individual "
    "would evacuate during a wildfire based on their lifestyle, mobility patterns, and "
    "proximity to danger. Be logical and realistic when evaluating decisions.\n"
)


instruction0 = "Below is the individual's lifestyle and mobility pattern:"
instruction1_default = "Below is a description of the individual's neighborhood:"
instruction2 = "Disaster scenario:"
instruction3 = (
    "The individual may or may not choose to evacuate, depending on their proximity to the fire, mobility routines, lifestyle, and personality cues. "
    "Use all available information to make a realistic guess about whether they will evacuate. "
    "Explain your reasoning first. Then, on a new line, give your final answer in this exact format:\n"
    "'Final answer: Evacuate' or 'Final answer: Not evacuate'\n"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)


# ==========================================
#  LLMAgent
# ==========================================
class LLMAgent(Agent):
    def __init__(self, unique_id, model_mesa, agent_data, included_keys, llm_engine):
        super().__init__(unique_id, model_mesa)

        self.llm_engine = llm_engine  # 외부에서 주입
        self.uid = str(agent_data["persona_id"])
        self.agent_data = agent_data

        self.lifestyle = agent_data.get("output", "")
        self.home_fire_distance = float(agent_data["distance_to_fire_km"])
        self.fire_direction = agent_data.get("direction_from_fire", "")
        self.direction = self.fire_direction if self.fire_direction else "unknown direction"

        self.alert = agent_data.get("alert_level", "")
        self.geoid = agent_data.get("geoid", "")
        self.tract_id = agent_data.get("tract_id_x", "")

        self.included_keys = included_keys
        self.evacuation_status = False
        self.evacuation_step = None
        self.generated_response = ""

    # ==========================================
    # Prompt 생성
    # ==========================================
    def generate_prompt(self):

        # ---------------------------------------------------------
        # A1 block (distance)
        # ---------------------------------------------------------
        if "A1" in self.included_keys:
            dist = f"{self.home_fire_distance:.2f}"
            A1_block = f"[A1] The fire is {dist} km away. [/A1]"
        else:
            A1_block = ""

        # ---------------------------------------------------------
        # A5 block (direction)
        # ---------------------------------------------------------
        if "A5" in self.included_keys:
            A5_block = f"[A5] The home is to the {self.direction} of the fire. [/A5]"
        else:
            A5_block = ""

        # ---------------------------------------------------------
        base_fire_description = (
            "The Caldor Fire began on August 14, 2021, near Little Mountain between Omo Ranch and Grizzly Flats "
            "in El Dorado County, California. The fire ultimately burned more than 221,500 acres—about 346 square miles."
            "By early September, containment efforts improved, and evacuation orders were gradually lifted."
        )

        fire_movement = (
            "The Caldor Fire initially spread eastward from its ignition point on August 14, moving across forested terrain. "
            "On August 27, strong winds accelerated the fire’s advance toward north-east. " 
            "By August 30, the fire reached five miles of South Lake Tahoe which is further north-east. "
            "Continued firefighting efforts, including construction of extensive control lines, slowed the fire’s progress, and containment steadily increased over the following week."
        )

        # ---------------------------------------------------------
        # Neighborhood BEFORE (A2, A3, A4)
        # ---------------------------------------------------------
        neighborhood_before = []

        if "A2" in self.included_keys:
            pop_cat = self.agent_data.get("total_population_category", "unknown")
            neighborhood_before.append(f"[A2] This individual lives in a {pop_cat} population area. [/A2]")

        if "A3" in self.included_keys:
            income_cat = self.agent_data.get("median_income_category", "unknown")
            neighborhood_before.append(f"[A3] Their household income is considered {income_cat}. [/A3]")

        if "A4" in self.included_keys:
            white_cat = self.agent_data.get("white_population_category", "unknown")
            neighborhood_before.append(
                f"[A4] Their neighborhood has a {white_cat} proportion of white population. [/A4]"
            )

        # ---------------------------------------------------------
        # Neighborhood AFTER (A6 alert)
        # ---------------------------------------------------------
        neighborhood_after = []

        if "A6" in self.included_keys:
            alert_raw = str(self.alert).lower().strip()

            if alert_raw == "normal":
                neighborhood_after.append(
                    "[A6] There are currently no threats to life or property in this individual's home CBG. [/A6]"
                )
            elif alert_raw not in ["", "none"]:
                neighborhood_after.append(
                    f"[A6] This individual's home CBG has been alerted as {self.alert}. [/A6]"
                )

        # ---------------------------------------------------------
        # FINAL PROMPT ASSEMBLY
        # ---------------------------------------------------------
        user_content = (
            f"{instruction0}\n\n{self.lifestyle}\n\n"
            f"{instruction2}\n{base_fire_description}\n{fire_movement}\n\n"
            f"{instruction1_default}\n\n"
            f"{'\n'.join(neighborhood_before)}\n\n"
            f"{A1_block}\n"
            f"{A5_block}\n\n"
            f"{'\n'.join(neighborhood_after)}\n\n"
            f"{instruction3}"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    # ==========================================
    # LLM 호출 및 판단
    # ==========================================
    def check_evacuation(self):
        prompt = self.generate_prompt()
        lora = LoRARequest("adapter", 1, LORA_MODEL_PATH)
        
        print(f"\n================= SYSTEM MESSAGE =================\n")
        print(system_message)

        print(f"\n🔹 Prompt for Agent {self.uid}:\n{prompt}\n")

        t0 = time.perf_counter()
        output = self.llm_engine.generate(
            prompt, 
            sampling_params=sampling_params, 
            lora_request=lora,
            use_tqdm=False
        )[0]
        t1 = time.perf_counter()

        response = output.outputs[0].text if hasattr(output.outputs[0], "text") else ""
        self.generated_response = response

        print(f"🔹 Response:\n{response}\n⏱ Time: {t1 - t0:.2f}s")

        match = re.search(r"Final answer:\s*(Evacuate|Not evacuate)", response, re.IGNORECASE)
        final = match.group(1).strip().lower() if match else "unknown"

        self.evacuation_status = (final == "evacuate")

        if self.evacuation_status:
            print(f"🚨 Agent {self.uid} has evacuated!")
        else:
            print(f"✅ Agent {self.uid} stays.")

    def step(self):
        if self.evacuation_status:
            return

        self.check_evacuation()

        if self.evacuation_status:
            self.evacuation_step = self.model.current_step
            self.model.evacuated_agents.append(self)
            self.model.schedule.remove(self)
