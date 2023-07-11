import mesa

from agent import IceFisherAgent


class IceFishingModel(mesa.Model):
    def __init__(self, width=100, height=100, n_agents=5):
        self.n_agents = n_agents
        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.n_agents):
            a = IceFisherAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        self.schedule.step()

        # collect data
        # self.datacollector.collect(self)



