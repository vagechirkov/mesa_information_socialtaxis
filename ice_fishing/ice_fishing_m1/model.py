import mesa
import numpy as np
from mesa.space import MultiGrid

from .agent import IceFisherAgent, Fish
from .policy import RandomSearch


class IceFishingModel(mesa.Model):
    def __init__(self,
                 width: int = 100,
                 height: int = 100,
                 n_agents: int = 5,
                 max_fishing_time: int = 10,
                 fish_patch_size: int = 3):
        self.n_agents = n_agents
        self.fish_patch_size = fish_patch_size
        self.grid = MultiGrid(width, height, torus=False)
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "Total catch": lambda m: sum(
                    [a.total_catch for a in m.schedule.agents if isinstance(a, IceFisherAgent)])},
        )
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.n_agents):
            a = IceFisherAgent(i, self, RandomSearch(), "moving", max_fishing_time=max_fishing_time)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            # x = self.random.randrange(self.grid.width)
            # y = self.random.randrange(self.grid.height)
            x = width // 2
            y = height // 2
            self.grid.place_agent(a, (x, y))

        # add uniform circle of fish in the middle
        i = 0
        for x in range(width):
            for y in range(height):
                if (x - width // 2) ** 2 + (y - height // 2) ** 2 < self.fish_patch_size ** 2:
                    f = Fish(self.n_agents + i, self, catch_rate=0.7)
                    self.schedule.add(f)
                    self.grid.place_agent(f, (x, y))
                    i += 1

    def step(self):
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)

    def run_model(self, step_count: int = 100) -> None:
        for _ in range(step_count):
            self.step()
