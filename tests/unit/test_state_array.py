"""
Tests for StateArray functionality in utils.py
"""

import numpy as np
import pytest

from laser_measles.utils import StateArray


class TestStateArray:
    """Test cases for StateArray wrapper class."""

    def test_basic_creation(self):
        """Test basic StateArray creation and initialization."""
        data = np.zeros((3, 10))
        state_names = ["S", "I", "R"]
        states = StateArray(data, state_names=state_names)

        assert isinstance(states, np.ndarray)
        assert states.shape == (3, 10)
        assert states.state_names == ["S", "I", "R"]

    def test_attribute_access(self):
        """Test accessing states by attribute names."""
        data = np.zeros((4, 5))
        state_names = ["S", "E", "I", "R"]
        states = StateArray(data, state_names=state_names)

        # Test getter access
        assert np.array_equal(states.S, states[0])
        assert np.array_equal(states.E, states[1])
        assert np.array_equal(states.I, states[2])
        assert np.array_equal(states.R, states[3])

    def test_attribute_assignment(self):
        """Test assigning values through attribute access."""
        data = np.zeros((3, 10))
        state_names = ["S", "I", "R"]
        states = StateArray(data, state_names=state_names)

        # Test setter access
        states.S[:] = 1000
        states.I[:] = 50
        states.R[:] = 100

        assert np.all(states[0] == 1000)
        assert np.all(states[1] == 50)
        assert np.all(states[2] == 100)

    def test_slicing_operations(self):
        """Test that slicing works with attribute access."""
        data = np.random.rand(3, 10)
        state_names = ["S", "I", "R"]
        states = StateArray(data, state_names=state_names)

        # Test slicing with attributes
        patch_indices = [0, 2, 4]
        s_subset = states.S[patch_indices]
        i_subset = states.I[patch_indices]

        assert len(s_subset) == 3
        assert len(i_subset) == 3
        assert np.array_equal(s_subset, states[0, patch_indices])
        assert np.array_equal(i_subset, states[1, patch_indices])

    def test_numpy_operations(self):
        """Test that numpy operations work correctly."""
        data = np.ones((3, 5)) * 100
        state_names = ["S", "I", "R"]
        states = StateArray(data, state_names=state_names)

        # Test array operations
        total_pop = states.sum(axis=0)
        assert np.all(total_pop == 300)  # 100 + 100 + 100

        # Test prevalence calculation
        prevalence = states.I / total_pop
        assert np.all(prevalence == 1 / 3)  # 100 / 300

    def test_backward_compatibility(self):
        """Test that numeric indexing still works."""
        data = np.random.rand(4, 8)
        state_names = ["S", "E", "I", "R"]
        states = StateArray(data, state_names=state_names)

        # Both should give same results
        assert np.array_equal(states[0], states.S)
        assert np.array_equal(states[1], states.E)
        assert np.array_equal(states[2], states.I)
        assert np.array_equal(states[3], states.R)

    def test_invalid_attribute_access(self):
        """Test that invalid attribute names raise AttributeError."""
        data = np.zeros((3, 5))
        state_names = ["S", "I", "R"]
        states = StateArray(data, state_names=state_names)

        with pytest.raises(AttributeError):
            _ = states.X  # X is not a valid state name

        with pytest.raises(AttributeError):
            states.Y = 100  # Y is not a valid state name

    def test_different_state_configurations(self):
        """Test StateArray with different state configurations."""
        # Test SIR model (biweekly)
        sir_data = np.zeros((3, 10))
        sir_states = StateArray(sir_data, state_names=["S", "I", "R"])

        assert hasattr(sir_states, "S")
        assert hasattr(sir_states, "I")
        assert hasattr(sir_states, "R")
        assert not hasattr(sir_states, "E")

        # Test SEIR model (compartmental)
        seir_data = np.zeros((4, 10))
        seir_states = StateArray(seir_data, state_names=["S", "E", "I", "R"])

        assert hasattr(seir_states, "S")
        assert hasattr(seir_states, "E")
        assert hasattr(seir_states, "I")
        assert hasattr(seir_states, "R")

    def test_get_state_index(self):
        """Test the get_state_index utility method."""
        data = np.zeros((4, 5))
        state_names = ["S", "E", "I", "R"]
        states = StateArray(data, state_names=state_names)

        assert states.get_state_index("S") == 0
        assert states.get_state_index("E") == 1
        assert states.get_state_index("I") == 2
        assert states.get_state_index("R") == 3
        assert states.get_state_index("X") is None  # Invalid state

    def test_array_finalize(self):
        """Test that StateArray metadata is preserved during operations."""
        data = np.ones((3, 5))
        state_names = ["S", "I", "R"]
        states = StateArray(data, state_names=state_names)

        # Test that slicing preserves StateArray properties
        subset = states[:, :3]
        assert isinstance(subset, StateArray)
        assert subset.state_names == ["S", "I", "R"]

    def test_realistic_epidemiological_operations(self):
        """Test realistic epidemiological operations."""
        # Setup initial SEIR population
        num_patches = 10
        data = np.zeros((4, num_patches))
        state_names = ["S", "E", "I", "R"]
        states = StateArray(data, state_names=state_names)

        # Initialize with susceptible population
        initial_pop = np.random.randint(1000, 10000, num_patches)
        states.S[:] = initial_pop

        # Simulate some infections
        new_infections = np.random.randint(0, 50, num_patches)
        states.S -= new_infections
        states.E += new_infections

        # Check conservation of population
        total_pop = states.sum(axis=0)
        assert np.allclose(total_pop, initial_pop)

        # Test prevalence calculation
        prevalence = states.I / np.maximum(total_pop, 1)
        assert np.all(prevalence >= 0)
        assert np.all(prevalence <= 1)
