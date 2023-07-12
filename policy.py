from typing import Literal

import mesa


class Policy:
    def __init__(self):
        pass

    def select_action(self, **kwargs):
        """
        Select action based on the policy
        """
        raise NotImplementedError


class RandomSearch(Policy):
    def __init__(self):
        super().__init__()
        self.destination = None

    def select_action(self, model: mesa.Model, agent_position: tuple[int, int], last_catches: list[int]) -> Literal[
        "moving", "fishing"]:
        """
        Select action based on the policy
        """
        catch_rate = sum(last_catches[:50]) / len(last_catches[:50])

        if catch_rate < 1 / 3:
            x = model.random.randrange(model.grid.width)
            y = model.random.randrange(model.grid.height)
            self.destination = (x, y)
            return "moving"
        elif catch_rate < 2 / 3:
            # select neighbor cell
            neighbors = model.grid.get_neighborhood(agent_position, moore=True, include_center=False)
            x, y = model.random.choice(neighbors)
            self.destination = (x, y)
            return "moving"
        else:
            # continue fishing
            return "fishing"
