import numpy as np

from ice_fishing.ice_fishing_m1.belief import Belief


def test_init_belief():
    """
    Test the initialisation of the belief class
    """
    belief = Belief(100, 100)

    assert belief.catch_likelihood.shape == (100, 100), "The shape of the catch rate should be (100, 100)"
    assert belief.belief.shape == (100, 100), "The shape of the prior info should be (100, 100)"
    assert belief.social_likelihood.shape == (100, 100), "The shape of the social info should be (100, 100)"


def test_update_prior_info():
    """
    Test the update_prior_info method
    """
    belief = Belief(100, 100)

    belief.update_prior_belief(((50, 50),), radius=1)
    assert np.allclose(belief.belief.sum(), 1, atol=1e-5), "The sum of the prior info should be 1"
    assert np.allclose(belief.belief[50, 50], 0.2, atol=0.1), "The prior info should be a circle of 1 around 50, 50"

    belief.update_prior_belief(((50, 50), (70, 70)), radius=1)
    assert np.allclose(belief.belief.sum(), 1, atol=1e-5), "The sum of the prior info should be 1"
    assert np.allclose(belief.belief[50, 50], 0.1, atol=0.1), "The prior info should be a circle of 1 around 50, 50"


def test_update_social_likelihood():
    belief = Belief(100, 100)

    belief.update_social_likelihood(((50, 50),), radius=1)
    assert np.allclose(belief.social_likelihood[50, 50], 0.1, atol=1e-5)

    belief.update_social_likelihood(((50, 50), (70, 70)), radius=1)
    assert np.allclose(belief.social_likelihood[50, 50], 0.1, atol=1e-5)


def test_update_catch_likelihood():
    belief = Belief(100, 100)

    belief.update_catch_likelihood(((50, 50),), (0.5,), radius=1)
    assert np.allclose(belief.catch_likelihood[50, 50], 0.5, atol=1e-5)

    belief.update_catch_likelihood(((50, 50), (70, 70)), (0.4, 0.7), radius=1)
    assert np.allclose(belief.catch_likelihood[50, 50], 0.4, atol=1e-5)
    assert np.allclose(belief.catch_likelihood[70, 70], 0.7, atol=1e-5)
