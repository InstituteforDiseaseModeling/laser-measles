"""
Unit tests for event bus inheritance functionality.

Tests that verify the event bus functionality has been successfully moved
from ABMModel to BaseLaserModel and is properly inherited by child classes.
"""

import polars as pl
import pytest

from laser_measles.base import BaseLaserModel, BaseModelParams
from laser_measles.events import EventBus, ModelEvent


class MockModelParams(BaseModelParams):
    """Mock parameters for the test model."""
    
    @property
    def time_step_days(self) -> int:
        """Return the time step in days."""
        return 1


class MockComponent:
    """Mock component for testing."""
    
    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose
        self.initialized = False
    
    def __call__(self, model, tick: int) -> None:
        pass
    
    def _initialize(self, model) -> None:
        self.initialized = True


class MockModel(BaseLaserModel):
    """Test model that inherits from BaseLaserModel."""
    
    # Set scenario wrapper class to None to avoid auto-wrapping
    scenario_wrapper_class = None
    
    def __call__(self, model, tick: int) -> None:
        pass
    
    def _setup_components(self) -> None:
        # Call parent method to set up event capabilities
        super()._setup_components()
        # Add a mock component for testing
        if not hasattr(self, '_components') or not self._components:
            self.components = [MockComponent]


def test_event_bus_inheritance():
    """Test that event bus is available in child classes."""
    # Create test data
    scenario_data = pl.DataFrame({
        "pop": [1000, 2000],
        "lat": [0.0, 1.0],
        "lon": [0.0, 1.0]
    })
    
    params = MockModelParams(num_ticks=5, verbose=False)
    
    # Create test model
    model = MockModel(scenario_data, params, "test_model")
    
    # Verify event bus is available
    assert hasattr(model, 'event_bus'), "Model should have event_bus attribute"
    assert isinstance(model.event_bus, EventBus), "event_bus should be an EventBus instance"
    
    # Verify current_tick is available
    assert hasattr(model, 'current_tick'), "Model should have current_tick attribute"
    assert model.current_tick == 0, "current_tick should be initialized to 0"
    
    # Verify _create_model_event method is available
    assert hasattr(model, '_create_model_event'), "Model should have _create_model_event method"
    
    # Test creating a model event
    event = model._create_model_event('test_event', tick=1, test_data="hello")
    assert isinstance(event, ModelEvent), "Should create a ModelEvent"
    assert event.event_type == 'test_event', "Event type should match"
    assert event.tick == 1, "Event tick should match"
    assert event.data['test_data'] == 'hello', "Event data should match"
    assert event.data['model_name'] == 'test_model', "Model name should be included"


def test_abm_model_inheritance():
    """Test that ABMModel properly inherits event bus functionality."""
    from laser_measles.abm.model import ABMModel
    
    # Verify inheritance
    assert issubclass(ABMModel, BaseLaserModel), "ABMModel should inherit from BaseLaserModel"
    
    # Verify that ABMModel inherits event bus functionality from BaseLaserModel
    # by checking if the methods exist in the class or its parent classes
    assert '_create_model_event' in dir(ABMModel), "ABMModel should inherit _create_model_event from BaseLaserModel"
    
    # Test that an ABMModel instance has the event bus functionality
    import polars as pl
    from laser_measles.abm.params import ABMParams
    
    scenario_data = pl.DataFrame({
        "id": ["patch1"],
        "pop": [1000],
        "lat": [0.0],
        "lon": [0.0]
    })
    
    params = ABMParams(num_ticks=1, verbose=False)
    model = ABMModel(scenario_data, params, "test_abm")
    
    assert hasattr(model, 'event_bus'), "ABMModel instance should have event_bus"
    assert hasattr(model, 'current_tick'), "ABMModel instance should have current_tick"
    assert hasattr(model, '_create_model_event'), "ABMModel instance should have _create_model_event"


def test_base_model_event_capabilities():
    """Test that BaseLaserModel has all necessary event capabilities."""
    # Create test data
    scenario_data = pl.DataFrame({
        "pop": [1000],
        "lat": [0.0],
        "lon": [0.0]
    })
    
    params = MockModelParams(num_ticks=2, verbose=False)
    model = MockModel(scenario_data, params, "test_model")
    
    # Test that the model can emit events
    event_count = 0
    
    def event_handler(event):
        nonlocal event_count
        event_count += 1
    
    # Subscribe to test events
    model.event_bus.subscribe('test_event', event_handler)
    
    # Emit an event
    test_event = model._create_model_event('test_event', tick=1)
    model.event_bus.emit(test_event)
    
    # Verify event was received
    assert event_count == 1, "Event handler should have been called once"
    
    # Test event bus stats
    stats = model.event_bus.get_stats()
    assert stats['events_emitted'] == 1, "Should have emitted one event"
    assert stats['total_subscribers'] == 1, "Should have one subscriber"


def test_model_lifecycle_events():
    """Test that model lifecycle events are properly emitted."""
    # Create test data
    scenario_data = pl.DataFrame({
        "pop": [1000],
        "lat": [0.0],
        "lon": [0.0]
    })
    
    params = MockModelParams(num_ticks=1, verbose=False)
    model = MockModel(scenario_data, params, "test_model")
    
    # Ensure components are set up
    model._setup_components()
    
    # Track events
    events_received = []
    
    def event_handler(event):
        events_received.append(event.event_type)
    
    # Subscribe to model lifecycle events
    model.event_bus.subscribe('model_init', event_handler)
    model.event_bus.subscribe('tick_start', event_handler)
    model.event_bus.subscribe('tick_end', event_handler)
    model.event_bus.subscribe('model_complete', event_handler)
    
    # Run the model (this should emit lifecycle events)
    model.run()
    
    # Verify lifecycle events were emitted
    expected_events = ['model_init', 'tick_start', 'tick_end', 'model_complete']
    assert events_received == expected_events, f"Expected {expected_events}, got {events_received}"
