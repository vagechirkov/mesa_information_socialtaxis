from typing import Literal

import mesa
import numpy as np
import scipy

from .belief import Belief
from .utils.utils import catch_rate


class BaseIceFisher(mesa.Agent):
    def __init__(self,
                 unique_id: int,
                 model: mesa.Model,
                 state: Literal["moving", "fishing", "initial"] = "initial",
                 max_fishing_time: int = 10):
        super().__init__(unique_id, model)
        self.belief = Belief(model.grid.width, model.grid.height)
        self.state = state
        self.fishing_time = 0
        self.max_fishing_time = max_fishing_time
        self.total_catch = 0
        self.last_catches = []
        self.destination = None

        self.update_prior_belief([(model.grid.width // 2, model.grid.height // 2)])

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
            n_fish = self.model.resource_map[self.pos]
            p_catch = min(n_fish / self.model.fish_catch_threshold, 0.8)
            if np.random.rand() < p_catch:
                # fish is caught successfully
                self.total_catch += 1
                self.last_catches.append(1)
                # depletion of the resource
                self.model.resource_map[self.pos] -= 1
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

    def get_far_destination(self, radius=5) -> tuple[int, int]:
        # select random destination
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=radius)
        return self.model.random.choice(neighbors)

    def get_close_destination(self, radius=1) -> tuple[int, int]:
        # select random destination
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=radius)
        return self.model.random.choice(neighbors)

    def choose_action(self, rate: float = 0):
        if rate < 1 / 3:
            self.state = "moving"
            self.destination = self.get_far_destination()
        elif rate < 2 / 3:
            self.state = "moving"
            self.destination = self.get_close_destination()
        else:
            self.state = "fishing"

    def select_next_action(self):
        if not self._check_current_action_done():
            return

        if self.state == "initial":
            self.state = "moving"
            self.destination = self.get_far_destination()
        elif self.state == "moving":
            self.state = "fishing"
        elif self.state == "fishing":
            rate = catch_rate(self.last_catches, window_size=50)
            self.update_catch_rate_belief(rate)
            self.update_social_belief()
            self.choose_action(rate)
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

    def update_social_belief(self):
        # update social info
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=30)
        other_locs = [n.pos for n in neighbors if isinstance(n, BaseIceFisher)]
        self.belief.update_social_info(other_locs)

    def update_prior_belief(self, priors: list[tuple[int, int]]) -> None:
        # update prior info
        self.belief.update_prior_info(priors)

    def update_catch_rate_belief(self, rate: float) -> None:
        # update catch rate
        self.belief.update_catch_rate(self.pos, rate)


class ImitatorIceFisher(BaseIceFisher):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_far_destination(self, radius=5) -> tuple[int, int]:
        # select random destination
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=radius)

        # select agents with non-zero catch history
        neighbors_with_catches = [n for n in neighbors if hasattr(n, "last_catches") and sum(n.last_catches) > 0]

        # return random destination if no neighbors with non-zero catch history
        if len(neighbors_with_catches) == 0:
            neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=radius)
            return self.model.random.choice(neighbors)

        # select the most successful agent
        neighbors_with_catches = sorted(neighbors_with_catches, key=lambda x: sum(x.last_catches), reverse=True)

        # return the position of the most successful agent
        return neighbors_with_catches[0].pos


class BeliefFisher(BaseIceFisher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_action(self, rate: float = 0, temperature: float = 0.8):
        self.belief.softmax_weighted_belief(temperature=temperature)

        # select the most promising cell
        x, y = np.unravel_index(self.belief.softmax_belief.argmax(), self.belief.softmax_belief.shape)

        if (x, y) == self.pos:
            self.state = "fishing"
        else:
            self.state = "moving"
            self.destination = (x, y)
