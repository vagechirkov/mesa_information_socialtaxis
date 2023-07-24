import pytest
from ice_fishing.ice_fishing_m1.model import IceFishingModel


def tests_init():
    model = IceFishingModel()

    assert model.n_agents == 5
    assert model.grid.width == 100
    assert model.grid.height == 100
    assert model.running

    assert model.schedule.agents[0].state == "initial"

    # do one step
    model.step()

    assert model.schedule.agents[0].state == "moving"

    # do 10 steps
    for _ in range(10):
        model.step()

