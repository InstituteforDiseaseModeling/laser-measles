"""
Test script to verify LaserBaseModel cleanup functionality.
"""

import numpy as np
import polars as pl
import pytest

from laser_measles.abm import Model as ABMModel
from laser_measles.abm.params import ABMParams
from laser_measles.biweekly import BiweeklyModel
from laser_measles.biweekly import BiweeklyParams
from laser_measles.biweekly.base import BaseScenario


def create_test_scenario():
    """Create a small test scenario for biweekly model."""
    data = {
        "id": [f"patch_{i}" for i in range(10)],
        "pop": np.random.randint(1000, 5000, 10),
        "lat": np.random.uniform(-10, 10, 10),
        "lon": np.random.uniform(-10, 10, 10),
        "mcv1": np.random.uniform(0.5, 0.9, 10),
    }
    return pl.DataFrame(data)


def create_abm_scenario():
    """Create a small test scenario for ABM model."""
    data = {
        "id": [f"patch_{i}" for i in range(10)],
        "pop": np.random.randint(1000, 5000, 10),
        "lat": np.random.uniform(-10, 10, 10),
        "lon": np.random.uniform(-10, 10, 10),
        "mcv1": np.random.uniform(0.5, 0.9, 10),
    }
    return pl.DataFrame(data)


class TestModelCleanup:
    """Test cases for model cleanup functionality."""

    def test_biweekly_model_has_cleanup(self):
        """Test that BiweeklyModel has cleanup method."""
        scenario_df = create_test_scenario()
        scenario = BaseScenario(scenario_df)
        params = BiweeklyParams(start_time="2020-01", num_ticks=4)

        model = BiweeklyModel(scenario, params, name="test_biweekly")

        # Check that cleanup method exists
        assert hasattr(model, "cleanup"), "BiweeklyModel should have cleanup method"
        assert callable(model.cleanup), "cleanup should be callable"

        # Test cleanup doesn't raise errors
        model.cleanup()

    def test_generic_model_has_cleanup(self):
        """Test that Generic Model has cleanup method."""
        scenario = create_abm_scenario()
        abm_params = ABMParams(seed=42, num_ticks=4, start_time="2020-01")

        model = ABMModel(scenario, abm_params, name="test_generic")

        # Check that cleanup method exists
        assert hasattr(model, "cleanup"), "Generic Model should have cleanup method"
        assert callable(model.cleanup), "cleanup should be callable"

        # Test cleanup doesn't raise errors
        model.cleanup()

    def test_biweekly_model_cleanup_clears_patches(self):
        """Test that cleanup properly clears LaserFrame patches."""
        scenario_df = create_test_scenario()
        scenario = BaseScenario(scenario_df)
        params = BiweeklyParams(start_time="2020-01", num_ticks=4)

        model = BiweeklyModel(scenario, params, name="test_cleanup")

        # Verify patches exist before cleanup
        assert hasattr(model, "patches"), "Model should have patches"
        assert model.patches is not None, "Patches should not be None before cleanup"

        # Perform cleanup
        model.cleanup()

        # Verify patches are cleared
        assert model.patches is None, "Patches should be None after cleanup"

    def test_generic_model_cleanup_clears_laserframes(self):
        """Test that cleanup properly clears LaserFrame objects in generic model."""
        scenario = create_abm_scenario()
        abm_params = ABMParams(seed=42, num_ticks=4, start_time="2020-01")

        model = ABMModel(scenario, abm_params, name="test_cleanup")

        # Verify LaserFrames exist before cleanup
        assert hasattr(model, "patches"), "Model should have patches"
        assert hasattr(model, "people"), "Model should have people"
        assert model.patches is not None, "Patches should not be None before cleanup"
        assert model.people is not None, "People should not be None before cleanup"

        # Perform cleanup
        model.cleanup()

        # Verify LaserFrames are cleared
        assert model.patches is None, "Patches should be None after cleanup"
        assert model.people is None, "People should be None after cleanup"

    def test_cleanup_clears_components(self):
        """Test that cleanup properly clears component instances."""
        scenario_df = create_test_scenario()
        scenario = BaseScenario(scenario_df)
        params = BiweeklyParams(start_time="2020-01", num_ticks=4)

        model = BiweeklyModel(scenario, params, name="test_components")

        # Add some dummy components
        model._components = ["dummy1", "dummy2"]
        model.instances = ["instance1", "instance2"]
        model.phases = ["phase1", "phase2"]

        # Perform cleanup
        model.cleanup()

        # Verify components are cleared
        assert len(model._components) == 0, "Components should be cleared"
        assert len(model.instances) == 0, "Instances should be cleared"
        assert len(model.phases) == 0, "Phases should be cleared"

    def test_cleanup_clears_metrics(self):
        """Test that cleanup properly clears metrics."""
        scenario_df = create_test_scenario()
        scenario = BaseScenario(scenario_df)
        params = BiweeklyParams(start_time="2020-01", num_ticks=4)

        model = BiweeklyModel(scenario, params, name="test_metrics")

        # Add some dummy metrics
        model.metrics = [[1, 100, 200], [2, 150, 250]]

        # Perform cleanup
        model.cleanup()

        # Verify metrics are cleared
        assert len(model.metrics) == 0, "Metrics should be cleared"

    def test_cleanup_handles_missing_attributes_gracefully(self):
        """Test that cleanup handles missing attributes without errors."""
        scenario_df = create_test_scenario()
        scenario = BaseScenario(scenario_df)
        params = BiweeklyParams(start_time="2020-01", num_ticks=4)

        model = BiweeklyModel(scenario, params, name="test_missing")

        # Remove some attributes to test robustness
        delattr(model, "patches")

        # Cleanup should not raise errors even with missing attributes
        model.cleanup()  # Should not raise any exceptions

    def test_cleanup_multiple_calls(self):
        """Test that calling cleanup multiple times doesn't cause errors."""
        scenario_df = create_test_scenario()
        scenario = BaseScenario(scenario_df)
        params = BiweeklyParams(start_time="2020-01", num_ticks=4)

        model = BiweeklyModel(scenario, params, name="test_multiple")

        # Call cleanup multiple times - should not raise errors
        model.cleanup()
        model.cleanup()
        model.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
