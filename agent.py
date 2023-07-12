from typing import Literal

import mesa
import numpy as np

from policy import Policy


class IceFisherAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, policy: Policy,
                 state: Literal["moving", "fishing"] = "moving", max_fishing_time: int = 10):
        super().__init__(unique_id, model)
        self.state = state
        self.fishing_time = 0
        self.max_fishing_time = max_fishing_time
        self.total_catch = 0
        self.last_catches = []
        self.policy = policy
        self.destination = None

    def move(self, destination: tuple[int, int]):
        """
        Move agent one cell closer to the destination
        """
        x, y = self.pos
        dx, dy = destination
        if x < dx:
            x += 1
        elif x > dx:
            x -= 1
        if y < dy:
            y += 1
        elif y > dy:
            y -= 1
        self.model.grid.move_agent(self, (x, y))

    def fish(self):
        """
        Fish in the current cell
        """
        # increase fishing time
        self.fishing_time += 1

        # catch fish with probability p
        fish = [agent for agent in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(agent, Fish)]
        catch_rate = 0 if len(fish) == 0 else fish[0].catch_rate
        if np.random.rand() < catch_rate:
            # fish is caught successfully
            self.total_catch += 1
            self.last_catches.append(1)
        else:
            self.last_catches.append(0)

    def step(self):
        # if agent is not alone in the cell, move to empty cell
        if self.state == "moving":
            # move agent one cell closer to the selected destination
            if self.destination is not None:
                self.move(self.destination)

        else:  # fishing
            self.fish()

        if (self.state == "fishing" and self.fishing_time > self.max_fishing_time) or (
                self.pos == self.destination) or (self.destination is None):
            # update state
            self.state = self.policy.select_action(model=self.model, agent=self)
            self.destination = self.policy.destination
            # reset catch history
            self.last_catches = []


class Fish(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, catch_rate: float = 0.5):
        super().__init__(unique_id, model)
        self.catch_rate = catch_rate

    def step(self):
        pass
