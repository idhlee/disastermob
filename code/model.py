#model.py
import pandas as pd
from mesa import Model
from mesa.time import RandomActivation
from agent import LLMAgent
import datetime

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
        self.num_agents = len(self.df)
        self.current_step = 0
        self.schedule = RandomActivation(self)
        self.evacuated_agents = []

        self.agent_list = []
        for i, agent_data in enumerate(self.df.to_dict(orient="records")):
            agent = LLMAgent(i, self, agent_data, included_keys=self.included_keys)
            self.schedule.add(agent)
            self.agent_list.append(agent)

        self.fire_start_date = datetime.date(2021, 12, 30)
        self.fire_current_date = self.fire_start_date

    def step(self):
        self.current_step += 1
        self.fire_current_date = self.fire_start_date + datetime.timedelta(weeks=self.current_step - 1)
        self.x_th_day = ordinal(self.current_step)

        print(f"\n Model Step {self.current_step} Start ({self.fire_current_date.strftime('%B %d, %Y')})")
        self.schedule.step()
        print(" Model Step End")
