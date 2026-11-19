
"""LLM agent for wildfire evacuation inference with optional LoRA adapters."""

import re
import time

from mesa import Agent


def _is_missing(value):
    if value is None:
        return True
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _text(value, default="unknown"):
    return default if _is_missing(value) else str(value).strip()


class LLMAgent(Agent):
    def __init__(
        self,
        unique_id,
        model_mesa,
        agent_data,
        llm,
        tokenizer,
        sampling_params,
        prompt_config,
        included_keys=None,
        print_prompts=False,
        lora_request=None,
    ):
        super().__init__(unique_id, model_mesa)

        self.uid = _text(agent_data.get("persona_id"), str(unique_id))
        self.agent_data = agent_data
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.prompt_config = prompt_config
        self.included_keys = set(included_keys or [])
        self.print_prompts = print_prompts
        self.lora_request = lora_request

        self.lifestyle = _text(agent_data.get("persona_output"), default="")
        self.home_fire_distance = float(agent_data["distance_to_fire_km"])
        self.fire_direction = _text(
            agent_data.get("direction_from_fire"),
            default="unknown direction",
        )
        self.alert = _text(agent_data.get("alert_level"), default="")

        self.geoid = _text(agent_data.get("geoid"), default="")
        self.tract_id = _text(agent_data.get("tract_id"), default="")

        self.generated_response = ""
        self.final_decision = "unknown"
        self.evacuation_status = None
        self.evacuation_step = None
        self.inference_seconds = None

    def _build_lifestyle_block(self):
        instruction = self.prompt_config["instruction_lifestyle"]
        return f"{instruction} {self.lifestyle}"

    def _build_environment_block(self):
        keys = self.included_keys
        socio_parts = []
        fire_parts = []

        if "A2" in keys:
            pop = _text(
                self.agent_data.get("total_population_category"),
                default="unknown",
            )
            socio_parts.append(
                f"This individual lives in a {pop} population area. "
            )

        if "A3" in keys:
            income = _text(
                self.agent_data.get("median_income_category"),
                default="unknown",
            )
            socio_parts.append(
                f"Their household income is considered {income}. "
            )

        if "A4" in keys:
            white = _text(
                self.agent_data.get("white_population_category"),
                default="unknown",
            )
            socio_parts.append(
                f"Their neighborhood has a {white} proportion of white population. "
            )

        distance = self.home_fire_distance
        direction = self.fire_direction

        if "A1" in keys and "A5" in keys:
            fire_parts.append(
                f"The individual's home is located {distance:.2f} km "
                f"from the fire's ignition location, to the {direction} "
                "of that location. "
            )
        elif "A1" in keys:
            fire_parts.append(
                f"The individual's home is located {distance:.2f} km "
                "from the fire's ignition location. "
            )
        elif "A5" in keys:
            fire_parts.append(
                f"The individual's home is located to the {direction} "
                "of the fire's ignition location. "
            )

        if "A6" in keys:
            alert_raw = self.alert.lower().strip()

            if alert_raw == "normal":
                fire_parts.append(
                    "The initial emergency status for this individual's home CBG "
                    "is normal, with no evacuation warning or order recorded. "
                )
            elif alert_raw not in {"", "none", "nan", "null"}:
                fire_parts.append(
                    "The initial evacuation alert for this individual's home CBG "
                    f"is {self.alert}. "
                )

        instruction = self.prompt_config["instruction_environment"]
        socio_block = "".join(socio_parts)
        fire_block = "".join(fire_parts)

        return f"{instruction} {socio_block} {fire_block}"

    def _build_disaster_block(self):
        return (
            f"{self.prompt_config['instruction_disaster']} "
            f"{self.prompt_config['marshall_fire_description']}"
        )

    def build_messages(self):
        return [
            {
                "role": "system",
                "content": (
                    f"{self.prompt_config['system_message']} "
                    f"{self.prompt_config['instruction_task']}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{self._build_lifestyle_block()}\n\n"
                    f"{self._build_environment_block()}\n\n"
                    f"{self._build_disaster_block()}"
                ),
            },
        ]

    def generate_prompt(self):
        messages = self.build_messages()
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def parse_decision(response):
        matches = re.findall(
            r"Final\s+answer\s*:\s*(Not\s+evacuate|Evacuate)\b",
            response or "",
            flags=re.IGNORECASE,
        )

        if not matches:
            return "unknown"

        decision = re.sub(r"\s+", " ", matches[-1].strip().lower())
        return decision

    def check_evacuation(self):

        print(
            f"\n{'=' * 70}\n"
            f"Agent: {self.uid}\n"
            f"{'=' * 70}",
            flush=True,
        )

        prompt = self.generate_prompt()

        if self.print_prompts:
            print(
                f"\n========== PROMPT: {self.uid} ==========\n"
                f"{prompt}\n",
                flush=True,
            )

        start = time.perf_counter()

        request_output = self.llm.generate(
            prompt,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )[0]

        self.inference_seconds = time.perf_counter() - start

        if request_output.outputs:
            self.generated_response = request_output.outputs[0].text.strip()
        else:
            self.generated_response = ""

        print(
            f"\n========== RESPONSE: {self.uid} ==========\n"
            f"Response:\n{self.generated_response}\n"
            f"⏱ Time: {self.inference_seconds:.2f}s",
            flush=True,
        )

        self.final_decision = self.parse_decision(
            self.generated_response
        )

        if self.final_decision == "evacuate":
            self.evacuation_status = True
            self.evacuation_step = self.model.current_step
            self.model.evacuated_agents.append(self)
            self.model.schedule.remove(self)

        elif self.final_decision == "not evacuate":
            self.evacuation_status = False

        else:
            self.evacuation_status = None

        print(
            f"Final decision: {self.final_decision.upper()}\n"
            f"Evacuation status: {self.evacuation_status}\n"
            f"{'=' * 70}",
            flush=True,
        )

    def step(self):
        self.check_evacuation()
