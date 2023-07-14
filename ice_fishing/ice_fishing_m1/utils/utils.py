from scipy.stats import multivariate_normal

import numpy as np


def catch_rate(last_catches: list, window_size: int = 50) -> float:
    """
    Estimate the catch rate based on the last catches
    """
    n_catches = sum(last_catches[:window_size])
    time_window = min(len(last_catches), window_size)
    return n_catches / time_window if time_window > 0 else 0.0


def gaussian_resource_map(width: int, height: int, mean: tuple[float, float], cov_val: tuple[float, float],
                          random_seed=42, max_value=1) -> np.ndarray:
    """
    Create a gaussian resource map
    """
    # Initializing the covariance matrix
    cov = np.array([[1, cov_val[0]], [cov_val[1], 1]])

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix
    distr = multivariate_normal(cov=cov, mean=mean, seed=random_seed)

    # Generating a meshgrid complacent with
    # the 3-sigma boundary
    mean_1, mean_2 = mean[0], mean[1]
    sigma_1, sigma_2 = cov[0, 0], cov[1, 1]

    x = np.linspace(-3 * sigma_1, 3 * sigma_1, num=width)
    y = np.linspace(-3 * sigma_2, 3 * sigma_2, num=height)
    X, Y = np.meshgrid(x, y)

    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])

    normalized_pdf = (pdf / np.max(pdf)) * max_value
    return normalized_pdf

