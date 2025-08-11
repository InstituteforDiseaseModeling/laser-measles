#!/usr/bin/env python3
"""
Test script for the ABM event system.

This script creates a simple ABM model with vital dynamics and death monitoring
components to demonstrate the event system functionality.
"""

import polars as pl
import numpy as np
from laser_measles.abm.model import ABMModel
from laser_measles.abm.params import ABMParams
from laser_measles.abm.components import (
    VitalDynamicsProcess, 
    VitalDynamicsParams,
    DeathMonitorTracker,
    DeathMonitorParams,
    DeathAwareComponent
)


def create_simple_scenario():
    """Create a simple scenario for testing."""
    return pl.DataFrame({
        'pop': [1000, 800, 1200],  # Population in each patch
        'lat': [40.7, 40.8, 40.9],  # Latitude
        'lon': [-74.0, -74.1, -74.2],  # Longitude
        'id': ['patch_1', 'patch_2', 'patch_3']  # Patch IDs
    })


def test_has_subscribers():
    """Test the has_subscribers functionality."""
    print("Testing has_subscribers functionality")
    print("=" * 40)
    
    # Create a simple model for testing
    scenario = create_simple_scenario()
    params = ABMParams(num_ticks=1, verbose=False, show_progress=False, seed=12345)
    model = ABMModel(scenario, params, name="subscriber_test_model")
    
    # Create a test component
    vital_params = VitalDynamicsParams(crude_death_rate=0.1)  # Low rate for predictable testing
    
    class TestComponent(VitalDynamicsProcess):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=vital_params)
    
    model.components = [TestComponent]
    
    # Initialize the model by calling _initialize (this sets up the event bus)
    model._initialize()
    
    # Get the component instance
    test_component = model.instances[0]
    
    # Test 1: No subscribers initially
    print("Test 1: No subscribers for 'deaths' event")
    has_subs_before = test_component.has_subscribers('deaths')
    print(f"  has_subscribers('deaths'): {has_subs_before}")
    assert not has_subs_before, "Should have no subscribers initially"
    print("  ✓ Correctly reports no subscribers")
    
    # Test 2: Add a subscriber
    print("\nTest 2: Add subscriber for 'deaths' event")
    def dummy_handler(event):
        pass
    
    model.event_bus.subscribe('deaths', dummy_handler)
    has_subs_after = test_component.has_subscribers('deaths')
    print(f"  has_subscribers('deaths'): {has_subs_after}")
    assert has_subs_after, "Should have subscribers after subscribing"
    print("  ✓ Correctly reports subscribers exist")
    
    # Test 3: Check different event type
    print("\nTest 3: Check different event type 'births'")
    has_births_subs = test_component.has_subscribers('births')
    print(f"  has_subscribers('births'): {has_births_subs}")
    assert not has_births_subs, "Should have no subscribers for births"
    print("  ✓ Correctly reports no subscribers for different event type")
    
    # Test 4: Verify emit_event optimization
    print("\nTest 4: Test emit_event optimization")
    
    # Count events before
    stats_before = model.event_bus.get_stats()
    events_before = stats_before['events_emitted']
    
    # Emit event with no subscribers (should be optimized away)
    test_component.emit_event('nonexistent_event', data={'test': True})
    
    # Emit event with subscribers (should go through)
    test_component.emit_event('deaths', data={'test': True})
    
    stats_after = model.event_bus.get_stats()
    events_after = stats_after['events_emitted']
    
    events_emitted = events_after - events_before
    print(f"  Events emitted: {events_emitted}")
    assert events_emitted == 1, f"Should have emitted exactly 1 event, got {events_emitted}"
    print("  ✓ Only emitted event with subscribers")
    
    # Cleanup
    model.cleanup()
    print("\nhas_subscribers test passed!")
    print()


def test_event_system():
    """Test the event system with a simple simulation."""
    print("Testing ABM Event System")
    print("=" * 50)
    
    # Create scenario and parameters
    
    scenario = create_simple_scenario()
    params = ABMParams(
        num_ticks=10,  # Short simulation
        verbose=True,
        show_progress=False,  # Disable progress bar for cleaner output
        seed=12345
    )
    
    print(f"Scenario: {len(scenario)} patches with {scenario['pop'].sum()} total population")
    print(f"Simulation: {params.num_ticks} ticks")
    print()
    
    # Create model
    model = ABMModel(scenario, params, name="event_test_model")
    
    # Setup components with vital dynamics and event monitoring
    # Override the default death rate to make sure we see some deaths
    vital_params = VitalDynamicsParams(
        crude_death_rate=200.0  # Very high death rate for testing (200 per 1000 per year)
    )
    
    death_monitor_params = DeathMonitorParams(
        track_death_locations=True,
        track_death_states=True,
        verbose_deaths=True
    )
    
    # Create component classes with parameters
    class ConfiguredVitalDynamics(VitalDynamicsProcess):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=vital_params)
    
    class ConfiguredDeathMonitor(DeathMonitorTracker):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=death_monitor_params)
    
    # Add components to model
    model.components = [
        ConfiguredVitalDynamics,
        ConfiguredDeathMonitor,
        DeathAwareComponent
    ]
    
    print("Components added:")
    for i, comp in enumerate(model.components, 1):
        print(f"  {i}. {comp.__name__}")
    print()
    
    # Run the simulation
    print("Running simulation...")
    print("-" * 30)
    model.run()
    print("-" * 30)
    print("Simulation completed!")
    print()
    
    # Check event system statistics
    stats = model.event_bus.get_stats()
    print("Event System Statistics:")
    print(f"  Events emitted: {stats['events_emitted']}")
    print(f"  Total subscribers: {stats['total_subscribers']}")
    print(f"  Active event types: {stats['active_event_types']}")
    print(f"  Dispatch errors: {stats['dispatch_errors']}")
    print()
    
    # Get death monitor results
    death_monitor = None
    death_aware = None
    
    for instance in model.instances:
        if isinstance(instance, DeathMonitorTracker):
            death_monitor = instance
        elif isinstance(instance, DeathAwareComponent):
            death_aware = instance
    
    if death_monitor:
        summary = death_monitor.get_death_summary()
        print("Death Monitor Summary:")
        print(f"  Total deaths: {summary['total_deaths']}")
        print(f"  Death events: {summary['num_death_events']}")
        if summary['deaths_by_patch']:
            print(f"  Deaths by patch: {summary['deaths_by_patch']}")
        if summary['deaths_by_state']:
            print(f"  Deaths by state: {summary['deaths_by_state']}")
        print(f"  Recent death indices: {len(death_monitor.get_recent_deaths())}")
        print(f"  All death indices: {len(death_monitor.get_all_deaths())}")
        print()
    
    if death_aware:
        print("Death Aware Component:")
        print(f"  Death reactions: {death_aware.get_reaction_count()}")
        print()
    
    # Verify the event system worked
    print("Event System Verification:")
    if stats['events_emitted'] > 0:
        print("  ✓ Events were emitted")
    else:
        print("  ✗ No events were emitted")
    
    if stats['total_subscribers'] > 0:
        print("  ✓ Components subscribed to events")
    else:
        print("  ✗ No components subscribed to events")
    
    if death_monitor and death_monitor.total_deaths > 0:
        print("  ✓ Death events were processed")
    else:
        print("  ✗ No death events were processed")
    
    if death_aware and death_aware.get_reaction_count() > 0:
        print("  ✓ Components reacted to events")
    else:
        print("  ✗ No components reacted to events")
    
    print()
    print("Test completed successfully!")
    
    # Cleanup
    model.cleanup()
    
    return None


if __name__ == "__main__":
    # Run the has_subscribers test first
    test_has_subscribers()
    
    # Run the main event system test
    model, death_monitor, death_aware = test_event_system()
    
    # Optional: Interactive exploration
    print("\nComponents available for inspection:")
    print("  - model: ABMModel instance")
    print("  - death_monitor: DeathMonitorTracker instance")  
    print("  - death_aware: DeathAwareComponent instance")
    print("\nExample usage:")
    print("  death_monitor.get_death_summary()")
    print("  model.event_bus.get_stats()")