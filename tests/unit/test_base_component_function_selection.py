"""
Unit tests for BaseComponent function selection integration.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from laser_measles.abm.params import ABMParams
from laser_measles.base import BaseComponent


def numpy_test_func(x, y):
    """Simple numpy test function."""
    return x + y


def numba_test_func(x, y):
    """Simple numba test function."""
    return x * y


class TestBaseComponentFunctionSelection:
    """Test BaseComponent function selection integration."""

    def test_select_function_with_params(self):
        """Test select_function method with model parameters."""
        # Create mock model with parameters
        mock_model = MagicMock()
        mock_model.params = ABMParams(num_ticks=100, use_numba=False)

        # Create component
        component = BaseComponent(mock_model)

        # Test function selection
        selected = component.select_function(numpy_test_func, numba_test_func)
        assert selected == numpy_test_func

    def test_select_function_with_numba_enabled(self):
        """Test select_function method with numba enabled."""
        # Create mock model with numba enabled
        mock_model = MagicMock()
        mock_model.params = ABMParams(num_ticks=100, use_numba=True)

        # Create component
        component = BaseComponent(mock_model)

        # Mock numba availability
        with patch("laser_measles.utils._check_numba_available", return_value=True):
            with patch("laser_measles.utils._get_numba_preference", return_value=True):
                selected = component.select_function(numpy_test_func, numba_test_func)
                assert selected == numba_test_func

    def test_select_function_fallback_no_numba(self):
        """Test select_function method falls back to numpy when numba unavailable."""
        # Create mock model with numba enabled
        mock_model = MagicMock()
        mock_model.params = ABMParams(num_ticks=100, use_numba=True)

        # Create component
        component = BaseComponent(mock_model)

        # Mock numba unavailability
        with patch("laser_measles.utils._check_numba_available", return_value=False):
            with pytest.warns(UserWarning, match="Numba is not available"):  # noqa: PT031
                selected = component.select_function(numpy_test_func, numba_test_func)
                assert selected == numpy_test_func

    def test_select_function_no_use_numba_param(self):
        """Test select_function method when model has no use_numba parameter."""
        # Create mock model without use_numba parameter
        mock_model = MagicMock()
        mock_model.params = MagicMock()
        del mock_model.params.use_numba  # Remove the attribute

        # Create component
        component = BaseComponent(mock_model)

        # Test function selection (should default to True)
        with patch("laser_measles.utils._check_numba_available", return_value=True):
            with patch("laser_measles.utils._get_numba_preference", return_value=True):
                selected = component.select_function(numpy_test_func, numba_test_func)
                assert selected == numba_test_func
