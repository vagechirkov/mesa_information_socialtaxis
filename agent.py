from typing import Literal

import mesa
import numpy as np

from policy import Policy


class IceFisherAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, policy: Policy,
                 state: Literal["moving", "fishing"] = "moving"):
        super().__init__(unique_id, model)
        self.state = state
        self.policy = policy
        self.belief = np.zeros((self.model.grid.width, self.model.grid.height))

    def step(self):
        # identify action based on the policy
        # aciton =  self.policy.select_action(self.belief)

        # if agent is not alone in the cell, move to empty cell
        pass


class FishAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model):
        super().__init__(unique_id, model)

    def step(self):
        pass
