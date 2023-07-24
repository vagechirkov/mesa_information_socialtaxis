import pytest

from ice_fishing.ice_fishing_m1.utils.utils import generate_resource_map, mean_catch_ratio


@pytest.mark.parametrize("cluster_std", [0.1, 0.5, 1.0])
def test_generate_resource_map(cluster_std):
    """
    Test the generate_resource_map function
    """
    # test the function with default parameters
    resource_map = generate_resource_map(100, 100, cluster_std=cluster_std, n_samples=2000)

    assert resource_map.shape == (100, 100), "The shape of the resource map should be (100, 100)"
    assert resource_map.sum() == 2000, "The number of fish should be equal to n_samples"
    assert resource_map.min() >= 0, "The minimum value of the resource map should be greater than 0"


def test_mean_catch_ratio():
    result = mean_catch_ratio([1, 1, 1], 10)
    assert result == 0.1
