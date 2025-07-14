"""Tests for compartmental SIA calendar component."""
# ruff: noqa: DTZ001, RUF100, PT012, PT011

from datetime import datetime
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from laser_measles.compartmental import BaseScenario
from laser_measles.compartmental import CompartmentalModel
from laser_measles.compartmental import CompartmentalParams
from laser_measles.compartmental.components import SIACalendarParams
from laser_measles.compartmental.components import SIACalendarProcess


@pytest.fixture
def mock_scenario():
    """Create a mock scenario for testing."""
    data = {
        "id": ["NG:KN:001", "NG:KN:002", "NG:KD:001", "NG:KD:002"],
        "pop": [1000, 1500, 2000, 800],
        "lat": [12.0, 12.1, 11.9, 12.2],
        "lon": [8.5, 8.6, 8.4, 8.7],
        "mcv1": [0.8, 0.75, 0.85, 0.9],
    }
    df = pl.DataFrame(data)
    return BaseScenario(df)


@pytest.fixture
def mock_model_params():
    """Create mock model parameters."""
    return CompartmentalParams(num_ticks=100, start_time="2023-01")


@pytest.fixture
def mock_sia_schedule():
    """Create a mock SIA schedule."""
    return pl.DataFrame({"id": ["NG:KN", "NG:KD", "NG:KN"], "date": ["2023-01-10", "2023-01-15", "2023-01-25"]})


@pytest.fixture
def mock_model(mock_scenario, mock_model_params):
    """Create a mock compartmental model."""
    model = CompartmentalModel(mock_scenario, mock_model_params)
    return model


class TestSIACalendarParams:
    """Test SIACalendarParams class."""

    def test_default_params(self, mock_sia_schedule):
        """Test default parameter values."""
        params = SIACalendarParams(sia_schedule=mock_sia_schedule)
        assert params.sia_efficacy == 0.9
        assert params.aggregation_level == 3
        assert params.date_column == "date"
        assert params.group_column == "id"

    def test_custom_params(self, mock_sia_schedule):
        """Test custom parameter values."""
        params = SIACalendarParams(
            sia_schedule=mock_sia_schedule, sia_efficacy=0.8, aggregation_level=2, date_column="schedule_date", group_column="region_id"
        )
        assert params.sia_efficacy == 0.8
        assert params.aggregation_level == 2
        assert params.date_column == "schedule_date"
        assert params.group_column == "region_id"

    def test_efficacy_bounds(self, mock_sia_schedule):
        """Test SIA efficacy bounds validation."""
        # Valid bounds
        SIACalendarParams(sia_schedule=mock_sia_schedule, sia_efficacy=0.0)
        SIACalendarParams(sia_schedule=mock_sia_schedule, sia_efficacy=1.0)

        # Invalid bounds should raise validation error
        with pytest.raises(ValueError):
            SIACalendarParams(sia_schedule=mock_sia_schedule, sia_efficacy=-0.1)
        with pytest.raises(ValueError):
            SIACalendarParams(sia_schedule=mock_sia_schedule, sia_efficacy=1.1)


class TestSIACalendarProcess:
    """Test SIACalendarProcess class."""

    def test_initialization(self, mock_model, mock_sia_schedule):
        """Test component initialization."""
        # Test with aggregation level 2 to group by state
        params = SIACalendarParams(sia_schedule=mock_sia_schedule, aggregation_level=2)
        component = SIACalendarProcess(mock_model, params=params)

        assert len(component.node_mapping) == 2  # NG:KN and NG:KD
        assert "NG:KN" in component.node_mapping
        assert "NG:KD" in component.node_mapping
        assert len(component.node_mapping["NG:KN"]) == 2  # Two nodes in KN
        assert len(component.node_mapping["NG:KD"]) == 2  # Two nodes in KD

    def test_initialization_without_params(self, mock_model):
        """Test initialization fails without parameters."""
        with pytest.raises(ValueError, match="SIACalendarParams must be provided"):
            SIACalendarProcess(mock_model)

    def test_date_parsing(self, mock_model, mock_sia_schedule):
        """Test date parsing functionality."""
        params = SIACalendarParams(sia_schedule=mock_sia_schedule)
        component = SIACalendarProcess(mock_model, params=params)

        # Test YYYY-MM-DD format
        expected_date = datetime(2023, 1, 1)
        assert component.start_date == expected_date

        # Test tick to date conversion
        assert component._tick_to_date(0) == expected_date
        assert component._tick_to_date(10) == expected_date + timedelta(days=10)

    def test_date_parsing_month_format(self, mock_scenario, mock_sia_schedule):
        """Test date parsing with YYYY-MM format."""
        params = CompartmentalParams(
            num_ticks=100,
            start_time="2023-01",  # Month format
        )
        model = CompartmentalModel(mock_scenario, params)
        sia_params = SIACalendarParams(sia_schedule=mock_sia_schedule)
        component = SIACalendarProcess(model, params=sia_params)

        expected_date = datetime(2023, 1, 1)
        assert component.start_date == expected_date

    def test_invalid_date_format(self, mock_scenario, mock_sia_schedule):
        """Test invalid date format handling."""
        # First test invalid format in model initialization
        with pytest.raises(ValueError):
            params = CompartmentalParams(num_ticks=100, start_time="invalid-date", beta=0.5, sigma=1.0 / 8.0, gamma=1.0 / 5.0)
            CompartmentalModel(mock_scenario, params)

    def test_parameter_validation(self, mock_model):
        """Test parameter validation."""
        # Test missing required columns
        invalid_schedule = pl.DataFrame({"wrong_column": ["value"]})
        params = SIACalendarParams(sia_schedule=invalid_schedule)

        with pytest.raises(ValueError, match="sia_schedule must contain columns"):
            SIACalendarProcess(mock_model, params=params)

        # Test invalid aggregation level
        valid_schedule = pl.DataFrame({"id": ["test"], "date": ["2023-01-01"]})
        params = SIACalendarParams(sia_schedule=valid_schedule, aggregation_level=0)

        with pytest.raises(ValueError, match="aggregation_level must be at least 1"):
            SIACalendarProcess(mock_model, params=params)

    def test_sia_implementation(self, mock_model, mock_sia_schedule):
        """Test SIA implementation functionality."""
        params = SIACalendarParams(sia_schedule=mock_sia_schedule, sia_efficacy=1.0, aggregation_level=2)
        component = SIACalendarProcess(mock_model, params=params)

        # Initialize model states
        np.random.seed(42)  # For reproducible testing

        # Set initial susceptible population
        initial_susceptible = mock_model.patches.states.S.copy()
        initial_recovered = mock_model.patches.states.R.copy()

        # Run component at tick 10 (2023-01-10) - should trigger KN SIA
        component(mock_model, tick=10)

        # Check that susceptibles decreased and recovered increased for KN nodes
        kn_indices = component.node_mapping["NG:KN"]
        assert (mock_model.patches.states.S[kn_indices] <= initial_susceptible[kn_indices]).all()
        assert (mock_model.patches.states.R[kn_indices] >= initial_recovered[kn_indices]).all()

        # Check that SIA was marked as implemented
        assert "NG:KN_2023-01-10 00:00:00" in component.implemented_sias

    def test_sia_not_implemented_twice(self, mock_model, mock_sia_schedule):
        """Test that SIAs are not implemented twice."""
        params = SIACalendarParams(sia_schedule=mock_sia_schedule, sia_efficacy=1.0, aggregation_level=2)
        component = SIACalendarProcess(mock_model, params=params)

        np.random.seed(42)

        # Run component twice at the same tick
        component(mock_model, tick=10)
        states_after_first = mock_model.patches.states.S.copy()

        component(mock_model, tick=10)
        states_after_second = mock_model.patches.states.S.copy()

        # States should be identical (no double vaccination)
        assert np.array_equal(states_after_first, states_after_second)

    def test_multiple_sias_different_dates(self, mock_model, mock_sia_schedule):
        """Test multiple SIAs at different dates."""
        params = SIACalendarParams(sia_schedule=mock_sia_schedule, sia_efficacy=1.0, aggregation_level=2)
        component = SIACalendarProcess(mock_model, params=params)

        np.random.seed(42)

        # Run at tick 10 (should trigger KN SIA)
        component(mock_model, tick=10)
        assert len(component.implemented_sias) == 1

        # Run at tick 15 (should trigger KD SIA)
        component(mock_model, tick=15)
        assert len(component.implemented_sias) == 2

        # Run at tick 25 (should trigger second KN SIA)
        component(mock_model, tick=25)
        assert len(component.implemented_sias) == 3

    def test_filtering_function(self, mock_model, mock_sia_schedule):
        """Test custom filtering function."""

        # Filter to only include KN nodes
        def filter_fn(x: str) -> bool:
            return "KN" in x

        params = SIACalendarParams(sia_schedule=mock_sia_schedule, filter_fn=filter_fn, aggregation_level=2)
        component = SIACalendarProcess(mock_model, params=params)

        # Should only have KN nodes in mapping
        assert len(component.node_mapping) == 1
        assert "NG:KN" in component.node_mapping
        assert "NG:KD" not in component.node_mapping

    def test_different_aggregation_levels(self, mock_model, mock_sia_schedule):
        """Test different aggregation levels."""
        # Test aggregation level 2 (country:state)
        params = SIACalendarParams(sia_schedule=mock_sia_schedule, aggregation_level=2)
        component = SIACalendarProcess(mock_model, params=params)

        # Should group by NG:KN and NG:KD
        assert len(component.node_mapping) == 2
        assert "NG:KN" in component.node_mapping
        assert "NG:KD" in component.node_mapping

    def test_get_sia_schedule(self, mock_model, mock_sia_schedule):
        """Test getting SIA schedule."""
        params = SIACalendarParams(sia_schedule=mock_sia_schedule)
        component = SIACalendarProcess(mock_model, params=params)

        schedule = component.get_sia_schedule()
        assert schedule.equals(mock_sia_schedule)

    def test_verbose_output(self, mock_model, mock_sia_schedule, capsys):
        """Test verbose output."""
        params = SIACalendarParams(sia_schedule=mock_sia_schedule, sia_efficacy=1.0, aggregation_level=2)
        component = SIACalendarProcess(mock_model, verbose=True, params=params)

        # Check initialization message
        captured = capsys.readouterr()
        assert "SIACalendar initialized with 2 groups" in captured.out

        # Run component to trigger SIA
        np.random.seed(42)
        component(mock_model, tick=10)

        # Check SIA implementation message
        captured = capsys.readouterr()
        assert "Implementing SIA for NG:KN" in captured.out
        assert "vaccinated" in captured.out

    def test_empty_sia_schedule(self, mock_model):
        """Test behavior with empty SIA schedule."""
        empty_schedule = pl.DataFrame({"id": [], "date": []}).with_columns([pl.col("id").cast(pl.String), pl.col("date").cast(pl.String)])

        params = SIACalendarParams(sia_schedule=empty_schedule)
        component = SIACalendarProcess(mock_model, params=params)

        # Should not crash and should not implement any SIAs
        initial_states = mock_model.patches.states.S.copy()
        component(mock_model, tick=10)
        final_states = mock_model.patches.states.S.copy()

        assert np.array_equal(initial_states, final_states)
        assert len(component.implemented_sias) == 0
