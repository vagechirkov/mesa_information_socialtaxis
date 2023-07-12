from typing import Literal

import mesa


class Policy:
    def __init__(self):
        self.destination = None

    def select_action(self, **kwargs):
        """
        Select action based on the policy
        """
        raise NotImplementedError


class RandomSearch(Policy):
    def __init__(self):
        super().__init__()

    def select_action(self, model: mesa.Model, agent: mesa.Agent) -> Literal["moving", "fishing"]:
        """
        Select action based on the policy
        """
        if agent.state == "moving":
            if agent.destination is None:
                # select random destination
                x = model.random.randrange(model.grid.width)
                y = model.random.randrange(model.grid.height)
                self.destination = (x, y)
            elif agent.pos != agent.destination:
                return "moving"
            elif agent.pos == agent.destination:
                # start fishing when destination is reached
                return "fishing"

        catch_rate = sum(agent.last_catches[:50]) / len(agent.last_catches[:50]) if len(
            agent.last_catches[:50]) > 0 else 0

        if catch_rate < 1 / 3:
            x = model.random.randrange(model.grid.width)
            y = model.random.randrange(model.grid.height)
            self.destination = (x, y)
            return "moving"
        elif catch_rate < 2 / 3:
            # select neighbor cell
            neighbors = model.grid.get_neighborhood(agent.pos, moore=True, include_center=False)
            x, y = model.random.choice(neighbors)
            self.destination = (x, y)
            return "moving"
        else:
            # continue fishing
            return "fishing"
