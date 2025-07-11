"""
Unit tests for function selection utilities.
"""

import os
from unittest.mock import patch

import pytest

from laser_measles.utils import _check_numba_available
from laser_measles.utils import _get_numba_preference
from laser_measles.utils import dual_implementation
from laser_measles.utils import select_implementation


def numpy_test_func(x, y):
    """Simple numpy test function."""
    return x + y


def numba_test_func(x, y):
    """Simple numba test function."""
    return x * y


class TestFunctionSelection:
    """Test function selection utilities."""

    def test_select_implementation_prefer_numpy(self):
        """Test selecting numpy implementation when use_numba=False."""
        selected = select_implementation(numpy_test_func, numba_test_func, use_numba=False)
        assert selected == numpy_test_func

    def test_select_implementation_prefer_numba(self):
        """Test selecting numba implementation when use_numba=True and numba available."""
        with patch("laser_measles.utils._check_numba_available", return_value=True):
            with patch("laser_measles.utils._get_numba_preference", return_value=True):
                selected = select_implementation(numpy_test_func, numba_test_func, use_numba=True)
                assert selected == numba_test_func

    def test_select_implementation_fallback_to_numpy(self):
        """Test fallback to numpy when numba not available."""
        with patch("laser_measles.utils._check_numba_available", return_value=False):
            with pytest.warns(UserWarning, match="Numba is not available"):  # noqa: PT031
                selected = select_implementation(numpy_test_func, numba_test_func, use_numba=True)
                assert selected == numpy_test_func

    def test_dual_implementation_decorator(self):
        """Test the dual implementation decorator."""
        test_func = dual_implementation(numpy_test_func, numba_test_func)

        # Test that both implementations are stored
        assert test_func.numpy_func == numpy_test_func
        assert test_func.numba_func == numba_test_func

        # Test calling with explicit use_numba=False
        result = test_func(2, 3, use_numba=False)
        assert result == 5  # numpy: 2 + 3

        # Test calling with use_numba=True and mocked numba availability
        with patch("laser_measles.utils._check_numba_available", return_value=True):
            with patch("laser_measles.utils._get_numba_preference", return_value=True):
                result = test_func(2, 3, use_numba=True)
                assert result == 6  # numba: 2 * 3

    def test_numba_availability_detection(self):
        """Test numba availability detection."""
        # This will test the actual numba availability
        available = _check_numba_available()
        assert isinstance(available, bool)

    def test_environment_variable_preference(self):
        """Test environment variable preference detection."""
        # Test default (should be True)
        with patch.dict(os.environ, {}, clear=True):
            assert _get_numba_preference()

        # Test explicit true values
        for val in ["true", "True", "1", "yes", "on"]:
            with patch.dict(os.environ, {"LASER_MEASLES_USE_NUMBA": val}):
                assert _get_numba_preference()

        # Test explicit false values
        for val in ["false", "False", "0", "no", "off"]:
            with patch.dict(os.environ, {"LASER_MEASLES_USE_NUMBA": val}):
                assert not _get_numba_preference()

    def test_environment_variable_override(self):
        """Test that environment variable overrides use_numba parameter."""
        with patch.dict(os.environ, {"LASER_MEASLES_USE_NUMBA": "false"}):
            selected = select_implementation(numpy_test_func, numba_test_func, use_numba=True)
            assert selected == numpy_test_func
