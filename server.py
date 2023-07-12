import mesa

from model import IceFishingModel


def draw_lake(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return

    if agent.state == "fishing":
        portrayal = {"Shape": "fisher-fishing.svg", "Layer": 0, "State": agent.state}
    elif agent.state == "moving":
        portrayal = {"Shape": "fisher-moving.svg", "Layer": 0, "State": agent.state}
    else:
        portrayal = {"Shape": "fisher-fishing.svg", "Layer": 0, "State": agent.state}


    portrayal["Color"] = ["#FF0000"]
    # portrayal["stroke_color"] = "#00FF00"

    # if agent.type == 0:
    #     portrayal["Color"] = ["#FF0000", "#FF9999"]
    #     portrayal["stroke_color"] = "#00FF00"
    # else:
    #     portrayal["Color"] = ["#0000FF", "#9999FF"]
    #     portrayal["stroke_color"] = "#000000"
    return portrayal


canvas_element = mesa.visualization.CanvasGrid(
    draw_lake, 20, 20, 500, 500)

model_params = {
    "height": 20,
    "width": 20,
    "n_agents": mesa.visualization.Slider("N agents", value=5, min_value=1, max_value=10, step=1),
}

server = mesa.visualization.ModularServer(
    IceFishingModel, [canvas_element], "Ice Fishing", model_params
)
