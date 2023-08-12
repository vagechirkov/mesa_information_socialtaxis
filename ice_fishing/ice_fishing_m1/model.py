import mesa
import numpy as np
from mesa.space import MultiGrid

from .agent_fisher import BaseIceFisher, ImitatorIceFisher, GreedyBayesFisher
from .agent_fish import Fish, BeliefHolderAgent
from .utils.utils import generate_resource_map, mean_catch_ratio


class IceFishingModel(mesa.Model):
    def __init__(self,
                 width: int = 100,
                 height: int = 100,
                 n_agents: int = 5,
                 max_fishing_time: int = 10,
                 fish_patch_std: int = 0.4,
                 fish_patch_n_samples: int = 100_000,
                 fish_catch_threshold: int = 100,
                 agent_model: str = "random",
                 server: bool = False):
        self.current_id = 0
        self.server = server
        self.n_agents = n_agents
        self.fish_catch_threshold = fish_catch_threshold
        self.grid = MultiGrid(width, height, torus=False)
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "Mean catch ratio": lambda m: mean_catch_ratio(
                    [agent.total_catch for agent in m.schedule.agents if isinstance(agent, BaseIceFisher)],
                    fish_patch_n_samples),
            },
        )
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # Create agents
        for _ in range(self.n_agents):
            if agent_model == "random":
                a = BaseIceFisher(self.next_id(), self, "initial", max_fishing_time=max_fishing_time)
            elif agent_model == "imitator":
                a = ImitatorIceFisher(self.next_id(), self, "initial", max_fishing_time=max_fishing_time)
            elif agent_model == "greedy_bayesian":
                a = GreedyBayesFisher(self.next_id(), self, "initial", max_fishing_time=max_fishing_time)
            else:
                raise ValueError(f"Unknown agent model: {agent_model}")
            self.schedule.add(a)

            # Add the agent to a random grid cell
            # x = self.random.randrange(self.grid.width)
            # y = self.random.randrange(self.grid.height)
            x = width // 2
            y = height // 2
            self.grid.place_agent(a, (x, y))

        self.resource_map = generate_resource_map(width, height, cluster_std=fish_patch_std,
                                                  n_samples=fish_patch_n_samples)

        if self.server:
            # this is needed to visualise the resource map in the server
            self._initialise_fish()
            self._update_fish_catch(p_catch=self.resource_map)
            self._update_belief_holder()

    def _initialise_fish(self):
        # add uniform circle of fish in the middle
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                f = Fish(self.next_id(), self, p_catch=0)
                self.schedule.add(f)
                self.grid.place_agent(f, (x, y))

                # add belief holder
                b = BeliefHolderAgent(self.next_id(), self)
                self.schedule.add(b)
                self.grid.place_agent(b, (x, y))

    def _update_fish_catch(self, p_catch: np.ndarray):
        for agent in self.schedule.agents:
            if isinstance(agent, Fish):
                agent.p_catch = p_catch[agent.pos]

    def _update_belief_holder(self):
        # find agent with id 0
        agent_0 = [agent for agent in self.schedule.agents if agent.unique_id == 1 and hasattr(agent, 'belief')][0]

        for agent in self.schedule.agents:
            if isinstance(agent, BeliefHolderAgent):
                agent.social_likelihood = agent_0.belief.social_likelihood[agent.pos]
                agent.catch_likelihood = agent_0.belief.catch_likelihood[agent.pos]
                agent.belief = agent_0.belief.belief[agent.pos]

    def step(self):
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)

        if self.server:
            # this is needed to visualise the resource map in the server
            self._update_fish_catch(p_catch=self.resource_map)
            self._update_belief_holder()

    def run_model(self, step_count: int = 100) -> None:
        for _ in range(step_count):
            self.step()
