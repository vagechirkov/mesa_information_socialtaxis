import mesa
import numpy as np
from mesa.space import MultiGrid

from .agent_fisher import RandomIceFisher
from .agent_fish import Fish
from .utils.utils import gaussian_resource_map


class IceFishingModel(mesa.Model):
    def __init__(self,
                 width: int = 100,
                 height: int = 100,
                 n_agents: int = 5,
                 max_fishing_time: int = 10,
                 fish_patch_size: int = 3):
        self.current_id = 0
        self.n_agents = n_agents
        self.fish_patch_size = fish_patch_size
        self.grid = MultiGrid(width, height, torus=False)
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "Total catch": lambda m: sum(
                    [a.total_catch for a in m.schedule.agents if isinstance(a, RandomIceFisher)])},
        )
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # Create agents
        for _ in range(self.n_agents):
            a = RandomIceFisher(self.next_id(), self, "initial", max_fishing_time=max_fishing_time)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            # x = self.random.randrange(self.grid.width)
            # y = self.random.randrange(self.grid.height)
            x = width // 2
            y = height // 2
            self.grid.place_agent(a, (x, y))
        self._initialise_fish()
        self._update_fish_catch(p_catch=gaussian_resource_map(width, height, (0, 0), (0.4, 0.4)))

    def _initialise_fish(self):
        # add uniform circle of fish in the middle
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                f = Fish(self.next_id(), self, p_catch=0)
                self.schedule.add(f)
                self.grid.place_agent(f, (x, y))

    def _update_fish_catch(self, p_catch: np.ndarray):
        for agent in self.schedule.agents:
            if isinstance(agent, Fish):
                agent.p_catch = p_catch[agent.pos[0], agent.pos[1]]

    def step(self):
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)

    def run_model(self, step_count: int = 100) -> None:
        for _ in range(step_count):
            self.step()
