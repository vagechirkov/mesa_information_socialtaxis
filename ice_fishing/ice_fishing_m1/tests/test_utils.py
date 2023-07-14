import pytest

from ice_fishing.ice_fishing_m1.utils.utils import generate_resource_map


@pytest.mark.parametrize("cluster_std", [0.1, 0.5, 1.0])
def test_generate_resource_map(cluster_std):
    """
    Test the generate_resource_map function
    """
    # test the function with default parameters
    resource_map = generate_resource_map(100, 100, cluster_std=cluster_std)

    # check that max is less than 1
    assert resource_map.max() < 1

    # check that min is greater than 0
    assert resource_map.min() >= 0
