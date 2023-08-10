import numpy as np

from ice_fishing.ice_fishing_m1.utils.discrete_bayes_filter import normalize, observation_update


def test_normalize():
    """
    Test the normalize function
    """
    unnormalized_pmf = np.array([1, 2, 3, 4, 5])
    normalized_pmf = normalize(unnormalized_pmf)

    assert np.isclose(normalized_pmf.sum(), 1, atol=1e-5), "The sum of the normalized pmf should be 1"


def test_observation_update():
    """
    Test the observation_update function
    """
    prior_belief = normalize(np.ones((100, 100)))
    observation = np.zeros((100, 100))
    observation[50, 50] = 1
    observation = normalize(observation)
    posterior_belief = observation_update(prior_belief, observation)

    assert np.isclose(posterior_belief.sum(), 1, atol=1e-5), "The sum of the posterior belief should be 1"
    assert np.isclose(posterior_belief[50, 50], 1, atol=1e-5), "The posterior belief should be 1 at 50, 50"

    observation = np.zeros((100, 100))
    observation[50, 50] = 0.5
    observation[60, 60] = 0.5
    observation = normalize(observation)
    posterior_belief = observation_update(prior_belief, observation)

    assert np.isclose(posterior_belief[50, 50], 0.5, atol=1e-5), "The posterior belief should be 0.5 at 50, 50"
    assert np.isclose(posterior_belief[60, 60], 0.5, atol=1e-5), "The posterior belief should be 0.5 at 60, 60"
