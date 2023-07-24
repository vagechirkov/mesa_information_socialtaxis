import mesa
import matplotlib as mpl

from .agent_fish import Fish, BeliefHolderAgent
from .agent_fisher import BaseIceFisher
from .model import IceFishingModel

cmap = mpl.colormaps['Blues']
norm = mpl.colors.Normalize(vmin=0, vmax=20)
m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

m_belief = mpl.cm.ScalarMappable(cmap=mpl.colormaps['Blues'], norm=mpl.colors.Normalize(vmin=0, vmax=.05))


def draw_grid(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return

    if isinstance(agent, Fish):
        color = mpl.colors.to_hex(m.to_rgba(agent.p_catch))

        portrayal = {"Color": color, "Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9}
        return portrayal

    if isinstance(agent, BeliefHolderAgent):
        return

    if agent.state == "fishing" and agent.fishing_time == 1:
        portrayal = {"Shape": "jackhammer.svg", "Layer": 1, "State": agent.state, "Catch": agent.total_catch}
    elif agent.state == "fishing":
        portrayal = {"Shape": "fisher-fishing.svg", "Layer": 1, "State": agent.state, "Catch": agent.total_catch}
    elif agent.state == "moving":
        portrayal = {"Shape": "fisher-moving.svg", "Layer": 1, "State": agent.state, "Catch": agent.total_catch}
    else:
        portrayal = {"Shape": "fisher-fishing.svg", "Layer": 1, "State": agent.state, "Catch": agent.total_catch}
    return portrayal


def draw_belief(agent, value: float):
    # draw circle for the agent with the id 1
    if isinstance(agent, BaseIceFisher) and agent.unique_id == 1:
        return {"Shape": "circle", "r": 0.5, "Filled": "true", "Color": "red", "Layer": 1}

    if not isinstance(agent, BeliefHolderAgent):
        return

    color = mpl.colors.to_hex(m_belief.to_rgba(value))
    portrayal = {"Color": color, "Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9}
    return portrayal


def draw_prior_belief(agent):
    return draw_belief(agent, agent.prior if hasattr(agent, "prior") else 0)


def draw_social_belief(agent):
    return draw_belief(agent, agent.social if hasattr(agent, "social") else 0)


def draw_catch_rate(agent):
    return draw_belief(agent, agent.catch_rate if hasattr(agent, "catch_rate") else 0)


def draw_softmax_belief(agent):
    return draw_belief(agent, agent.softmax_belief if hasattr(agent, "softmax_belief") else 0)


grid_size = 10
grid_canvas_size = 600

grid = mesa.visualization.CanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size)

# chart = mesa.visualization.ChartModule([{"Label": "Total catch", "Color": "Black"}],
#                                        canvas_height=100, canvas_width=1000,
#                                        data_collector_name='datacollector')

grid_belief_prior = mesa.visualization.CanvasGrid(
    draw_prior_belief, grid_size, grid_size, grid_canvas_size // 3, grid_canvas_size // 3)
grid_belief_social = mesa.visualization.CanvasGrid(
    draw_social_belief, grid_size, grid_size, grid_canvas_size // 3, grid_canvas_size // 3)
grid_belief_catch = mesa.visualization.CanvasGrid(
    draw_catch_rate, grid_size, grid_size, grid_canvas_size // 3, grid_canvas_size // 3)
grid_belief_softmax = mesa.visualization.CanvasGrid(
    draw_softmax_belief, grid_size, grid_size, grid_canvas_size // 3, grid_canvas_size // 3)

model_params = {
    "height": grid_size,
    "width": grid_size,
    "server": True,
    "agent_model": mesa.visualization.Choice("Agent model", value="weighted_information",
                                             choices=["random", "imitator", "weighted_information"]),
    "n_agents": mesa.visualization.Slider("N agents", value=5, min_value=1, max_value=10, step=1),
    "fish_patch_std": mesa.visualization.Slider("Patch std", value=0.4, min_value=0.01, max_value=1, step=0.1),
    "fish_patch_n_samples": mesa.visualization.NumberInput("Fish abondance", value=2000),
}

server = mesa.visualization.ModularServer(
    IceFishingModel, [grid, grid_belief_prior, grid_belief_catch, grid_belief_social, grid_belief_softmax],
    "Ice Fishing", model_params
)
