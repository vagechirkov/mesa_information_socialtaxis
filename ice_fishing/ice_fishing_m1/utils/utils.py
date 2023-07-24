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
                          centers: list[list[float]] = ((1, 1), (-1, 1), (1, -1), (-1, -1)),
                          random_seed: int = 42):
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_seed)
    x_edges = np.linspace(-2, 2, width + 1)
    y_edges = np.linspace(-2, 2, height + 1)
    fish_count, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=(x_edges, y_edges), density=False)

    # add missing samples to the random places
    n_missing_samples = n_samples - np.sum(fish_count)
    for _ in range(n_missing_samples.astype(int)):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        fish_count[x, y] += 1

    # rescale to [0, max_value]
    # fish_map = (fish_count / np.max(fish_count)) * max_value
    fish_map = fish_count

    # # cut off values below min_value
    # density[density < min_value] = 0

    return fish_map


def mean_catch_ratio(agents_total_catch: list[int], total_number_of_fish: int) -> float:
    """
    Calculate the total catch ratio to evaluate the model for the given abundance of fish
    """
    return np.mean(agents_total_catch) / total_number_of_fish


def draw_circe_around_point(array: np.ndarray, x: int, y: int, radius: int = 3) -> np.ndarray:
    assert radius > 0, "Radius should be larger than 0"
    assert x < array.shape[0], "x should be smaller than the array width"
    assert y < array.shape[1], "y should be smaller than the array height"

    for xi in range(max(0, x - radius), min(array.shape[0], x + radius + 1)):
        for yi in range(max(0, y - radius), min(array.shape[1], y + radius + 1)):
            if (xi - x) ** 2 + (yi - y) ** 2 <= radius ** 2:
                array[xi, yi] = 1
    return array
