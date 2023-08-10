import numpy as np


def normalize(unnormalized_pmf: np.ndarray) -> np.ndarray:
    """
    Normalize the unnormalized probability mass function (pmf) to a proper probability distribution

    :param unnormalized_pmf: the unnormalized pmf
    :return: the normalized pmf
    """
    return unnormalized_pmf / unnormalized_pmf.sum()


def control_update():
    # we don't need this for the ice fishing problem because there is no uncertainty in the agent control
    return NotImplementedError


def observation_update(prior_belief: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """
    Update the prior belief with the observation
    """
    posterior_belief = prior_belief * observation
    return normalize(posterior_belief)


def discrete_bayes_filter(prior_belief: np.ndarray, observation: np.ndarray) -> np.ndarray:
    return observation_update(prior_belief, observation)
