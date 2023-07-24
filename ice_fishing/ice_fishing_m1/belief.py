import numpy as np
import scipy

from ice_fishing.ice_fishing_m1.utils.utils import draw_circe_around_point


class Belief:
    def __init__(self, width: int, height: int) -> None:
        self.catch_rate = np.zeros((width, height))
        self.prior_info = np.zeros((width, height))
        self.social_info = np.zeros((width, height))
        self.softmax_belief = np.zeros((width, height))

    def update_prior_info(self, locs: list[tuple[int, int]], radius: int = 3) -> None:
        self.prior_info = np.zeros_like(self.prior_info)
        for x, y in locs:
            # make a circle around prior center location
            self.prior_info = draw_circe_around_point(self.prior_info, x, y, radius)

        # make sure the prior info is proper probability distribution
        self.prior_info = self.prior_info / self.prior_info.sum()

    def update_social_info(self, other_locs: list[tuple[int, int]], radius: int = 3) -> None:
        self.social_info = np.zeros_like(self.social_info)
        for x, y in other_locs:
            # make a circle around the other agent
            self.social_info = draw_circe_around_point(self.social_info, x, y, radius)

        # make sure the social info is proper probability distribution
        self.social_info = self.social_info / self.social_info.sum()

    def update_catch_rate(self, loc: tuple[int, int], catch_rate: float) -> None:
        add_catch_rate = draw_circe_around_point(np.zeros_like(self.catch_rate), loc[0], loc[1], radius=3)
        self.catch_rate += add_catch_rate * catch_rate
        # make sure the catch rate is proper probability distribution
        self.catch_rate = self.catch_rate / self.catch_rate.sum() if self.catch_rate.sum() != 0 else self.catch_rate

    def softmax_weighted_belief(self, weights: np.ndarray = (1 / 3, 1 / 3, 1 / 3), temperature: float = 0.1) -> None:
        weighted_belief = weights[0] * self.prior_info + weights[1] * self.social_info + weights[2] * self.catch_rate

        # softmax weighted belief with temperature
        self.softmax_belief = weighted_belief  # scipy.special.softmax(weighted_belief / temperature)
