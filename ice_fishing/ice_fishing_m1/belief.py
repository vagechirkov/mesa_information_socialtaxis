import numpy as np
import scipy

from ice_fishing.ice_fishing_m1.utils.discrete_bayes_filter import normalize
from ice_fishing.ice_fishing_m1.utils.utils import draw_circe_around_point as circle


class Belief:
    def __init__(self, width: int, height: int) -> None:
        self.catch_likelihood = np.zeros((width, height))
        self.social_likelihood = np.zeros((width, height))
        self.belief = np.zeros((width, height))

    def update_prior_belief(self,
                            locs: tuple[tuple[int, int], ...] = None,
                            radius: int = 3) -> None:
        self.belief = np.zeros_like(self.belief) + 1e-5  # add a small number to avoid 0 probability

        if locs is not None:
            for x, y in locs:
                # make a circle around prior center location
                self.belief += circle(np.zeros_like(self.belief), x, y, radius)
        self.belief = normalize(self.belief)

    def update_social_likelihood(self,
                                 other_locs: tuple[tuple[int, int], ...],
                                 radius: int = 5,
                                 weight: float = 0.1) -> None:
        # add a small number to avoid 0 probability
        self.social_likelihood = np.zeros_like(self.social_likelihood) + 1e-5
        for x, y in other_locs:
            # make a circle around the other agent
            self.social_likelihood += circle(np.zeros_like(self.social_likelihood), x, y, radius) * weight

    def update_catch_likelihood(self,
                                loc: tuple[tuple[int, int], ...],
                                catch_rates: tuple[float, ...],
                                radius: int = 3) -> None:
        self.catch_likelihood = np.zeros_like(self.catch_likelihood) + 1e-5  # add a small number to avoid 0 probability

        for (x, y), catch_rate in zip(loc, catch_rates):
            self.catch_likelihood += circle(np.zeros_like(self.catch_likelihood), x, y, radius=radius) * catch_rate

    def update_belief(self,
                      measures_social_loc: tuple[tuple[int, int], ...],
                      measures_catch_loc: tuple[tuple[int, int], ...],
                      measures_catch_value: tuple[float, ...]) -> None:
        self.update_social_likelihood(measures_social_loc)
        self.update_catch_likelihood(measures_catch_loc, measures_catch_value)
        self.belief = normalize(self.belief * self.social_likelihood * self.catch_likelihood)
