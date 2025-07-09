"""
Tests for the create_component function.
"""

from typing import Any

import pytest

from laser_measles.base import BaseComponent
from laser_measles.base import BaseLaserModel
from laser_measles.components import create_component


# Create a simple test component
class MockComponent(BaseComponent):
    def __init__(self, model: BaseLaserModel, verbose: bool = False, test_param: int = 42):
        super().__init__(model, verbose)
        self.test_param = test_param

    def initialize(self, model: BaseLaserModel) -> None:
        print(f"Initializing {self.__class__.__name__} with test_param={self.test_param}")

    def __call__(self, model: Any, tick: int) -> None:
        print(f"Executing {self.__class__.__name__} at tick {tick}")


# Mock model for testing
class MockModel:
    def __init__(self):
        self.params = type("Params", (), {"verbose": False})()


@pytest.fixture
def mock_model():
    """Fixture to provide a mock model for testing."""
    return MockModel()


@pytest.fixture
def component_factory():
    """Fixture to provide a component factory for testing."""
    return create_component(MockComponent, test_param=100)


def test_component_factory_creation(component_factory):
    """Test that the component factory is created correctly."""
    # Test that the factory is callable
    assert callable(component_factory)

    # Test string representation contains component class name
    factory_str = str(component_factory)
    assert "MockComponent" in factory_str

    # Test repr contains component class name
    factory_repr = repr(component_factory)
    assert "MockComponent" in factory_repr


def test_component_factory_call(component_factory, mock_model):
    """Test that the component factory can create component instances."""
    # Create component instance
    component_instance = component_factory(mock_model, verbose=True)

    # Test that the instance is of the correct type
    assert isinstance(component_instance, MockComponent)

    # Test that the component has the expected attributes
    assert hasattr(component_instance, "test_param")
    assert component_instance.test_param == 100

    # Test that the component has the expected methods
    assert hasattr(component_instance, "initialize")
    assert callable(component_instance)
    assert callable(component_instance.initialize)
    assert callable(component_instance)


def test_component_methods(component_factory, mock_model):
    """Test that component methods work correctly."""
    component_instance = component_factory(mock_model, verbose=True)

    # Test initialize method (should not raise an exception)
    try:
        component_instance.initialize(mock_model)
    except Exception as e:
        pytest.fail(f"initialize method raised an exception: {e}")

    # Test __call__ method (should not raise an exception)
    try:
        component_instance(mock_model, 1)
    except Exception as e:
        pytest.fail(f"__call__ method raised an exception: {e}")


def test_component_verbose_parameter(component_factory, mock_model):
    """Test that the verbose parameter is passed correctly."""
    # Test with verbose=True
    component_instance = component_factory(mock_model, verbose=True)
    assert component_instance.verbose is True

    # Test with verbose=False
    component_instance = component_factory(mock_model, verbose=False)
    assert component_instance.verbose is False


def test_component_default_parameters():
    """Test that default parameters work correctly."""
    # Create factory with default test_param
    default_factory = create_component(MockComponent)
    mock_model = MockModel()

    component_instance = default_factory(mock_model)
    assert component_instance.test_param == 42  # Default value
