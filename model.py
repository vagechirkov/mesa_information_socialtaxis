import mesa
from mesa.space import MultiGrid

from agent import IceFisherAgent
from policy import Policy


class IceFishingModel(mesa.Model):
    def __init__(self, width: int = 100, height: int = 100, n_agents: int = 5):
        self.n_agents = n_agents
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.n_agents):
            a = IceFisherAgent(i, self, Policy(), "moving")
            self.schedule.add(a)

            # Add the agent to a random grid cell
            # x = self.random.randrange(self.grid.width)
            # y = self.random.randrange(self.grid.height)
            x = width // 2
            y = 0
            self.grid.place_agent(a, (x, y))

    def step(self):
        self.schedule.step()

        # collect data
        # self.datacollector.collect(self)
