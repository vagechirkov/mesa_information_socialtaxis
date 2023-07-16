from scipy.stats import multivariate_normal

import numpy as np
from sklearn.datasets import make_blobs


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


def generate_resource_map(width: int, height: int,
                          max_value: float = 0.8,
                          cluster_std: float = 0.4,
                          n_samples: int = 100_000,
                          centers: list[list[float]] = ((1, 1), (-1, 1), (1, -1)),
                          random_seed: int = 42):
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_seed)
    x_edges = np.linspace(-2, 2, width + 1)
    y_edges = np.linspace(-2, 2, height + 1)
    density, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=(x_edges, y_edges), density=False)

    # rescale to [0, max_value]
    density = (density / np.max(density)) * max_value

    # # cut off values below min_value
    # density[density < min_value] = 0

    return density
