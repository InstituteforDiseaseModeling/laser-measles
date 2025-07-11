"""
Tests for auto-wrapping functionality of polars DataFrames in model constructors.
"""

import polars as pl
import pytest

from laser_measles.abm import ABMModel
from laser_measles.abm import ABMParams
from laser_measles.abm.base import BaseABMScenario
from laser_measles.biweekly import BiweeklyModel
from laser_measles.biweekly import BiweeklyParams
from laser_measles.biweekly.base import BaseBiweeklyScenario
from laser_measles.compartmental import CompartmentalModel
from laser_measles.compartmental import CompartmentalParams
from laser_measles.compartmental.base import BaseCompartmentalScenario


@pytest.fixture
def test_scenario_df():
    """Create a simple test scenario DataFrame."""
    return pl.DataFrame(
        {
            "id": ["patch1", "patch2", "patch3"],
            "pop": [1000, 2000, 1500],
            "lat": [40.0, 41.0, 42.0],
            "lon": [-74.0, -73.0, -72.0],
            "mcv1": [0.8, 0.9, 0.85],
        }
    )


def test_abm_auto_wrapping(test_scenario_df):
    """Test that ABM model auto-wraps DataFrames correctly."""
    params = ABMParams(num_ticks=10, verbose=False)
    model = ABMModel(test_scenario_df, params)

    # Check that scenario is wrapped in BaseABMScenario
    assert isinstance(model.scenario, BaseABMScenario)


def test_biweekly_auto_wrapping(test_scenario_df):
    """Test that Biweekly model auto-wraps DataFrames correctly."""
    params = BiweeklyParams(num_ticks=10, verbose=False)
    model = BiweeklyModel(test_scenario_df, params)

    # Check that scenario is wrapped in BaseBiweeklyScenario
    assert isinstance(model.scenario, BaseBiweeklyScenario)


def test_compartmental_auto_wrapping(test_scenario_df):
    """Test that Compartmental model auto-wraps DataFrames correctly."""
    params = CompartmentalParams(num_ticks=10, verbose=False)
    model = CompartmentalModel(test_scenario_df, params)

    # Check that scenario is wrapped in BaseCompartmentalScenario
    assert isinstance(model.scenario, BaseCompartmentalScenario)


def test_existing_scenario_preserved(test_scenario_df):
    """Test that existing scenario objects are not re-wrapped."""
    # Create wrapped scenario manually
    wrapped_scenario = BaseABMScenario(test_scenario_df)

    # Create model with wrapped scenario (should not re-wrap)
    params = ABMParams(num_ticks=10, verbose=False)
    model = ABMModel(wrapped_scenario, params)

    # Check that the same object is preserved
    assert model.scenario is wrapped_scenario


def test_scenario_data_access(test_scenario_df):
    """Test that wrapped scenarios can access data correctly."""
    params = ABMParams(num_ticks=10, verbose=False)
    model = ABMModel(test_scenario_df, params)

    # Check that we can access the underlying DataFrame
    assert len(model.scenario) == 3
    assert model.scenario["pop"].sum() == 4500
    assert model.scenario["lat"].mean() == 41.0
