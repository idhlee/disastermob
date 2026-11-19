"""Mesa model for LLM-based wildfire evacuation simulation."""

from mesa import Model
from mesa.time import RandomActivation

from agent import LLMAgent


class LLMModel(Model):
    """One-shot initial-stage evacuation simulation for a dataframe of personas."""

    def __init__(
        self,
        df,
        llm,
        tokenizer,
        sampling_params,
        prompt_config,
        included_keys,
        seed=0,
        print_prompts=False,
        lora_request=None,
    ):
        super().__init__()
        try:
            self.random.seed(seed)
        except AttributeError:
            pass

        self.df = df.reset_index(drop=True)
        self.num_agents = len(self.df)
        self.current_step = 0
        self.schedule = RandomActivation(self)
        self.evacuated_agents = []
        self.agent_list = []

        for i, agent_data in enumerate(self.df.to_dict(orient="records")):
            agent = LLMAgent(
                unique_id=i,
                model_mesa=self,
                agent_data=agent_data,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                prompt_config=prompt_config,
                included_keys=included_keys,
                print_prompts=print_prompts,
                lora_request=lora_request,
            )
            self.schedule.add(agent)
            self.agent_list.append(agent)

    def step(self):
        self.current_step += 1
        self.schedule.step()
