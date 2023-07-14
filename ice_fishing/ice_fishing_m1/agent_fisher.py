from typing import Literal

import mesa
import numpy as np

from .agent_fish import Fish
from .utils.utils import catch_rate


class RandomIceFisher(mesa.Agent):
    def __init__(self,
                 unique_id: int,
                 model: mesa.Model,
                 state: Literal["moving", "fishing", "initial"] = "initial",
                 max_fishing_time: int = 10):
        super().__init__(unique_id, model)
        self.state = state
        self.fishing_time = 0
        self.max_fishing_time = max_fishing_time
        self.total_catch = 0
        self.last_catches = []
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
        # first time no fishing, only drilling
        if self.fishing_time > 0:
            # catch fish with probability p
            fish = [agent for agent in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(agent, Fish)]
            if np.random.rand() < 0 if len(fish) == 0 else fish[0].catch_rate:
                # fish is caught successfully
                self.total_catch += 1
                self.last_catches.append(1)
            else:
                self.last_catches.append(0)

        # increase fishing time
        self.fishing_time += 1

    def _check_fishing_done(self) -> bool:
        """
        Check if fishing is done
        """
        fishing_state = self.state == "fishing"
        fishing_done = self.fishing_time == self.max_fishing_time
        return fishing_state and fishing_done

    def _check_moving_done(self) -> bool:
        """
        Check if moving is done
        """
        moving_state = self.state == "moving"
        destination_reached = self.pos == self.destination
        return moving_state and destination_reached

    def _check_destination_none(self) -> bool:
        """
        Check if destination is None
        """
        moving_state = self.state == "moving"
        destination_none = self.destination is None
        return moving_state and destination_none

    def _check_current_action_done(self) -> bool:
        """
        Check if the current action is done
        """
        return self._check_fishing_done() or \
            self._check_moving_done() or \
            self._check_destination_none() or \
            self.state == "initial"

    def _clean_up_previous_state(self):
        # reset catch history
        self.last_catches = []
        self.fishing_time = 0

    def get_random_destination(self, radius=5) -> tuple[int, int]:
        # select random destination
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=radius)
        return self.model.random.choice(neighbors)

    def select_next_action(self):
        if not self._check_current_action_done():
            return

        if self.state == "initial":
            self.state = "moving"
            self.destination = self.get_random_destination(radius=5)
        elif self.state == "moving":
            self.state = "fishing"
        elif self.state == "fishing":
            rate = catch_rate(self.last_catches, window_size=50)
            if rate < 1 / 3:
                self.state = "moving"
                self.destination = self.get_random_destination(radius=5)
            elif rate < 2 / 3:
                self.state = "moving"
                self.destination = self.get_random_destination(radius=1)
            else:
                self.state = "fishing"
        else:
            raise ValueError("Unknown state")

        self._clean_up_previous_state()

    def step(self):
        # select the next action if the current is done
        self.select_next_action()

        # if agent is not alone in the cell, move to empty cell
        if self.state == "moving":
            self.move(self.destination)
        elif self.state == "fishing":
            self.fish()
        else:
            raise ValueError("Unknown state")