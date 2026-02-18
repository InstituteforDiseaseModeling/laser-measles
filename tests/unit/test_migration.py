"""
Convention is:
A 2D matrix, M, representing the interaction network, where each element network[i, j] corresponds to the flow from node i to node j.
"""

import polars as pl
import pytest

from laser.measles.mixing.gravity import GravityMixing
from laser.measles.mixing.gravity import GravityParams


def mean_field_2pop_model():
    scenario = pl.DataFrame({"pop": [1000, 2000], "lat": [0, 0], "lon": [0, 1]})

    # Basic gravity model
    params = GravityParams(a=1, b=1, k=0.15, c=0)  # mean field model
    mixer = GravityMixing(scenario, params)

    return mixer


def mean_field_3pop_model():
    scenario = pl.DataFrame({"pop": [1000, 2000, 3000], "lat": [0, 0, 0], "lon": [0, 1, 2]})

    # Basic gravity model
    params = GravityParams(a=1, b=1, k=0.15, c=0)  # mean field model
    mixer = GravityMixing(scenario, params)

    return mixer


@pytest.mark.parametrize("mixer", [mean_field_3pop_model, mean_field_2pop_model])
def test_mean_field_migration_elements(mixer):
    mig_mat = mixer().migration_matrix
    # compare elements of the migration matrix
    assert mig_mat[0, 1] > mig_mat[1, 0], "Per agent probability of mixing from small to large is not greater than from large to small"


@pytest.mark.parametrize("mixer", [mean_field_3pop_model, mean_field_2pop_model])
def test_mean_field_migration_normalization(mixer):
    # assert normalization was correct
    assert mixer().params.k == (mixer().trips_out_of().sum() / mixer().scenario["pop"].sum())


@pytest.mark.parametrize("mixer_func", [mean_field_3pop_model, mean_field_2pop_model])
def test_mean_field_mixing_elements(mixer_func):
    mixer = mixer_func()
    mixing_mat = mixer.mixing_matrix
    # compare elements of the mixing matrix
    assert mixing_mat[0, 1] > mixing_mat[1, 0], (
        "Per agent probability of mixing from small to large is not greater than from large to small"
    )


if __name__ == "__main__":
    pytest.main([__file__ + "::test_mean_field_migration_elements", "-v", "-s"])
