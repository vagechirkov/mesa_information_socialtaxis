import numpy as np

from ice_fishing.ice_fishing_m1.belief import Belief


def test_init_belief():
    """
    Test the initialisation of the belief class
    """
    belief = Belief(100, 100)

    assert belief.catch_rate.shape == (100, 100), "The shape of the catch rate should be (100, 100)"
    assert belief.prior_info.shape == (100, 100), "The shape of the prior info should be (100, 100)"
    assert belief.social_info.shape == (100, 100), "The shape of the social info should be (100, 100)"


def test_update_prior_info():
    """
    Test the update_prior_info method
    """
    belief = Belief(100, 100)

    belief.update_prior_info([(50, 50)], radius=1)

    assert belief.prior_info.sum() == 5, "The prior info should be a circle of 1 around 50, 50"

    belief.update_prior_info([(50, 50), (70, 70)], radius=1)

    assert belief.prior_info.sum() == 10, "The prior info should be a circle of 1 around 50, 50 and 70, 70"


def test_softmax_weighted_belief():
    """
    Test the softmax_weighted_belief method
    """
    belief = Belief(100, 100)

    belief.update_prior_info([(50, 50)], radius=1)
    belief.update_social_info([(50, 50)], radius=1)
    belief.update_catch_rate((50, 50), 1)

    belief.softmax_weighted_belief()

    # check if the sum of the softmax belief is 1 with tolerance of 1e-5
    assert np.isclose(belief.softmax_belief.sum(), 1, atol=1e-5), "The sum of the softmax belief should be 1"
