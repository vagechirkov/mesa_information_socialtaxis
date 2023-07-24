import numpy as np

from ice_fishing.ice_fishing_m1.utils.utils import draw_circe_around_point


class Belief:
    def __init__(self, width: int, height: int) -> None:
        self.catch_rate = np.zeros((width, height))
        self.prior_info = np.zeros((width, height))
        self.social_info = np.zeros((width, height))

    def update_prior_info(self, locs: list[tuple[int, int]], radius: int = 3) -> None:
        self.prior_info = np.zeros_like(self.prior_info)
        for x, y in locs:
            # make a circle around prior center location
            self.prior_info = draw_circe_around_point(self.prior_info, x, y, radius)

    def update_social_info(self, other_locs: list[tuple[int, int]], radius: int = 3) -> None:
        self.social_info = np.zeros_like(self.social_info)
        for x, y in other_locs:
            # make a circle around the other agent
            self.social_info = draw_circe_around_point(self.social_info, x, y)

    def update_catch_rate(self, loc: tuple[int, int], catch_rate: float) -> None:
        self.catch_rate[loc] = catch_rate
