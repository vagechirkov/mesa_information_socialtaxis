import mesa

from .agent_fish import Fish
from .model import IceFishingModel


def draw_grid(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return

    if isinstance(agent, Fish):
        # generate color based on the continuous catch rate light gray
        color = "#%02x%02x%02x" % (
            int(255 * agent.catch_rate), int(255 * agent.catch_rate), int(255 * agent.catch_rate))

        portrayal = {"Color": color, "Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9}
        return portrayal

    if agent.state == "fishing" and agent.fishing_time == 1:
        portrayal = {"Shape": "jackhammer.svg", "Layer": 1, "State": agent.state, "Catch": agent.total_catch}
    elif agent.state == "fishing":
        portrayal = {"Shape": "fisher-fishing.svg", "Layer": 1, "State": agent.state, "Catch": agent.total_catch}
    elif agent.state == "moving":
        portrayal = {"Shape": "fisher-moving.svg", "Layer": 1, "State": agent.state, "Catch": agent.total_catch}
    else:
        portrayal = {"Shape": "fisher-fishing.svg", "Layer": 1, "State": agent.state, "Catch": agent.total_catch}
    return portrayal


grid_size = 30
grid_canvas_size = 600

grid = mesa.visualization.CanvasGrid(
    draw_grid, grid_size, grid_size, grid_canvas_size, grid_canvas_size)

chart = mesa.visualization.ChartModule([{"Label": "Total catch", "Color": "Black"}],
                                       canvas_height=100, canvas_width=1000,
                                       data_collector_name='datacollector')

model_params = {
    "height": grid_size,
    "width": grid_size,
    "n_agents": mesa.visualization.Slider("N agents", value=5, min_value=1, max_value=10, step=1),
    "fish_patch_size": mesa.visualization.Slider("Patch size", value=5, min_value=1, max_value=grid_size // 2, step=1),
}

server = mesa.visualization.ModularServer(
    IceFishingModel, [grid, grid], "Ice Fishing", model_params
)
