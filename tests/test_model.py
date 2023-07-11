import pytest


def tests_init():
    from model import IceFishers
    model = IceFishers()

    assert model.n_agents == 5
    assert model.grid.width == 100
    assert model.grid.height == 100
    assert model.running == True
