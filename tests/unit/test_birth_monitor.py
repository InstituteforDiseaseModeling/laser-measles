#!/usr/bin/env python3
"""
Test script for the BirthMonitorTracker component.

This script creates a simple ABM model with vital dynamics and birth monitoring
components to demonstrate and validate the birth event system functionality.
"""

import polars as pl
import numpy as np
from laser_measles.abm.model import ABMModel
from laser_measles.abm.params import ABMParams
from laser_measles.abm.components import (
    VitalDynamicsProcess, 
    VitalDynamicsParams,
    BirthMonitorTracker,
    BirthMonitorParams,
    BirthAwareComponent
)


def create_simple_scenario():
    """Create a simple scenario for testing."""
    return pl.DataFrame({
        'pop': [1000, 800, 1200],  # Population in each patch
        'lat': [40.7, 40.8, 40.9],  # Latitude
        'lon': [-74.0, -74.1, -74.2],  # Longitude
        'id': ['patch_1', 'patch_2', 'patch_3']  # Patch IDs
    })


def test_birth_has_subscribers():
    """Test the has_subscribers functionality for birth events."""
    print("Testing birth event has_subscribers functionality")
    print("=" * 50)
    
    # Create a simple model for testing
    scenario = create_simple_scenario()
    params = ABMParams(num_ticks=1, verbose=False, show_progress=False, seed=12345)
    model = ABMModel(scenario, params, name="birth_subscriber_test_model")
    
    # Create a test component with births enabled
    vital_params = VitalDynamicsParams(
        crude_birth_rate=50.0,  # Enable births
        crude_death_rate=0.1    # Low death rate for predictable testing  
    )
    
    class TestComponent(VitalDynamicsProcess):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=vital_params)
    
    model.components = [TestComponent]
    
    # Initialize the model by calling _initialize (this sets up the event bus)
    model._initialize()
    
    # Get the component instance
    test_component = model.instances[0]
    
    # Test 1: No subscribers initially
    print("Test 1: No subscribers for 'births' event")
    has_subs_before = test_component.has_subscribers('births')
    print(f"  has_subscribers('births'): {has_subs_before}")
    assert not has_subs_before, "Should have no subscribers initially"
    print("  ✓ Correctly reports no subscribers")
    
    # Test 2: Add a subscriber
    print("\nTest 2: Add subscriber for 'births' event")
    def dummy_handler(event):
        pass
    
    model.event_bus.subscribe('births', dummy_handler)
    has_subs_after = test_component.has_subscribers('births')
    print(f"  has_subscribers('births'): {has_subs_after}")
    assert has_subs_after, "Should have subscribers after subscribing"
    print("  ✓ Correctly reports subscribers exist")
    
    # Test 3: Check different event type
    print("\nTest 3: Check different event type 'deaths'")
    has_deaths_subs = test_component.has_subscribers('deaths')
    print(f"  has_subscribers('deaths'): {has_deaths_subs}")
    assert not has_deaths_subs, "Should have no subscribers for deaths"
    print("  ✓ Correctly reports no subscribers for different event type")
    
    # Test 4: Verify emit_event optimization
    print("\nTest 4: Test emit_event optimization for births")
    
    # Count events before
    stats_before = model.event_bus.get_stats()
    events_before = stats_before['events_emitted']
    
    # Emit event with no subscribers (should be optimized away)
    test_component.emit_event('nonexistent_event', data={'test': True})
    
    # Emit event with subscribers (should go through)
    test_component.emit_event('births', data={'test': True})
    
    stats_after = model.event_bus.get_stats()
    events_after = stats_after['events_emitted']
    
    events_emitted = events_after - events_before
    print(f"  Events emitted: {events_emitted}")
    assert events_emitted == 1, f"Should have emitted exactly 1 event, got {events_emitted}"
    print("  ✓ Only emitted event with subscribers")
    
    # Cleanup
    model.cleanup()
    print("\nbirth has_subscribers test passed!")
    print()


def test_birth_monitor_system():
    """Test the birth monitoring system with a simple simulation."""
    print("Testing ABM Birth Monitor System")
    print("=" * 60)
    
    # Create scenario and parameters
    scenario = create_simple_scenario()
    params = ABMParams(
        num_ticks=15,  # Longer simulation to ensure births occur
        verbose=True,
        show_progress=False,  # Disable progress bar for cleaner output
        seed=12345
    )
    
    print(f"Scenario: {len(scenario)} patches with {scenario['pop'].sum()} total population")
    print(f"Simulation: {params.num_ticks} ticks")
    print()
    
    # Create model
    model = ABMModel(scenario, params, name="birth_test_model")
    
    # Setup components with vital dynamics and birth monitoring
    # Use high birth rate to ensure births occur during short simulation
    vital_params = VitalDynamicsParams(
        crude_birth_rate=100.0,  # High birth rate for testing (100 per 1000 per year)
        crude_death_rate=20.0    # Some deaths too for realistic simulation
    )
    
    birth_monitor_params = BirthMonitorParams(
        track_birth_locations=True,
        verbose_births=True
    )
    
    # Create component classes with parameters
    class ConfiguredVitalDynamics(VitalDynamicsProcess):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=vital_params)
    
    class ConfiguredBirthMonitor(BirthMonitorTracker):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=birth_monitor_params)
    
    # Add components to model
    model.components = [
        ConfiguredVitalDynamics,
        ConfiguredBirthMonitor,
        BirthAwareComponent
    ]
    
    print("Components added:")
    for i, comp in enumerate(model.components, 1):
        print(f"  {i}. {comp.__name__}")
    print()
    
    # Run the simulation
    print("Running simulation...")
    print("-" * 40)
    model.run()
    print("-" * 40)
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
    
    # Get birth monitor results
    birth_monitor = None
    birth_aware = None
    
    for instance in model.instances:
        if isinstance(instance, BirthMonitorTracker):
            birth_monitor = instance
        elif isinstance(instance, BirthAwareComponent):
            birth_aware = instance
    
    # Validate birth monitor results
    if birth_monitor:
        summary = birth_monitor.get_birth_summary()
        print("Birth Monitor Summary:")
        print(f"  Total births: {summary['total_births']}")
        print(f"  Birth events: {summary['num_birth_events']}")
        if summary['births_by_patch']:
            print(f"  Births by patch: {summary['births_by_patch']}")
        print(f"  Recent birth indices: {len(birth_monitor.get_recent_births())}")
        print(f"  All birth indices: {len(birth_monitor.get_all_births())}")
        print()
        
        # Additional validation
        recent_births = birth_monitor.get_recent_births()
        all_births = birth_monitor.get_all_births()
        
        print("Birth Data Validation:")
        print(f"  Recent births count: {len(recent_births)}")
        print(f"  Total unique births: {len(all_births)}")
        
        # Check that all recent births are in the all births set
        if recent_births:
            recent_in_all = all(idx in all_births for idx in recent_births)
            print(f"  Recent births in total set: {'✓' if recent_in_all else '✗'}")
        
        # Validate birth indices are reasonable (should be consecutive from some starting point)
        if all_births:
            min_idx = min(all_births)
            max_idx = max(all_births)
            print(f"  Birth index range: {min_idx} to {max_idx}")
            
        print()
    
    if birth_aware:
        print("Birth Aware Component:")
        print(f"  Birth reactions: {birth_aware.get_reaction_count()}")
        print()
    
    # Verify the birth monitoring system worked
    print("Birth Monitor System Verification:")
    if stats['events_emitted'] > 0:
        print("  ✓ Events were emitted")
    else:
        print("  ✗ No events were emitted")
    
    if stats['total_subscribers'] > 0:
        print("  ✓ Components subscribed to events")
    else:
        print("  ✗ No components subscribed to events")
    
    if birth_monitor and birth_monitor.total_births > 0:
        print("  ✓ Birth events were processed")
        
        # Check if births occurred in multiple ticks
        births_by_tick = birth_monitor.get_birth_summary()['births_by_tick']
        if len(births_by_tick) > 1:
            print("  ✓ Births occurred across multiple ticks")
        else:
            print("  ! Births occurred in limited ticks (may be expected)")
            
        # Check if births occurred in multiple patches
        births_by_patch = birth_monitor.get_birth_summary()['births_by_patch']
        if len(births_by_patch) > 1:
            print("  ✓ Births occurred in multiple patches")
        else:
            print("  ! Births occurred in limited patches (may be expected)")
            
    else:
        print("  ✗ No birth events were processed")
    
    if birth_aware and birth_aware.get_reaction_count() > 0:
        print("  ✓ Components reacted to birth events")
        
        # Validate reaction count matches birth events
        if (birth_monitor and 
            birth_aware.get_reaction_count() == birth_monitor.get_birth_summary()['num_birth_events']):
            print("  ✓ Reaction count matches birth events")
        else:
            print("  ! Reaction count differs from birth events")
    else:
        print("  ✗ No components reacted to birth events")
    
    # Validate birth/death balance for population dynamics
    if birth_monitor:
        # Get population before and after
        initial_pop = scenario['pop'].sum()
        final_states = model.patches.states.sum()
        final_pop = final_states.sum() if hasattr(final_states, 'sum') else final_states
        
        print(f"\nPopulation Dynamics:")
        print(f"  Initial population: {initial_pop}")
        print(f"  Final population: {final_pop}")
        print(f"  Net change: {final_pop - initial_pop}")
        print(f"  Total births tracked: {birth_monitor.total_births}")
    
    print()
    print("Birth monitor test completed successfully!")
    
    # Cleanup
    model.cleanup()
    
    return None


def test_birth_data_integrity():
    """Test the integrity of birth data tracking."""
    print("Testing Birth Data Integrity")
    print("=" * 40)
    
    # Create a smaller, controlled scenario
    scenario = pl.DataFrame({
        'pop': [500, 300],  # Smaller population for precise tracking
        'lat': [40.7, 40.8],
        'lon': [-74.0, -74.1], 
        'id': ['patch_1', 'patch_2']
    })
    
    params = ABMParams(
        num_ticks=5,  # Short simulation
        verbose=False,
        show_progress=False,
        seed=42  # Different seed for variety
    )
    
    model = ABMModel(scenario, params, name="birth_integrity_test")
    
    # Configure components
    vital_params = VitalDynamicsParams(
        crude_birth_rate=200.0,  # Very high rate to ensure births
        crude_death_rate=0.0     # No deaths to simplify tracking
    )
    
    birth_params = BirthMonitorParams(
        track_birth_locations=True,
        verbose_births=False  # Reduce output for cleaner test
    )
    
    class TestVitalDynamics(VitalDynamicsProcess):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=vital_params)
    
    class TestBirthMonitor(BirthMonitorTracker):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=birth_params)
    
    model.components = [TestVitalDynamics, TestBirthMonitor]
    
    print("Running integrity test simulation...")
    model.run()
    
    # Get the birth monitor
    birth_monitor = None
    for instance in model.instances:
        if isinstance(instance, BirthMonitorTracker):
            birth_monitor = instance
            break
    
    if birth_monitor:
        summary = birth_monitor.get_birth_summary()
        
        print("Data Integrity Checks:")
        print(f"  Total births: {summary['total_births']}")
        print(f"  Birth events: {summary['num_birth_events']}")
        
        # Check consistency between different metrics
        births_by_tick_sum = sum(summary['births_by_tick'].values())
        births_by_patch_sum = sum(summary['births_by_patch'].values())
        
        print(f"  Sum of births by tick: {births_by_tick_sum}")
        print(f"  Sum of births by patch: {births_by_patch_sum}")
        
        # Validate consistency
        consistency_checks = []
        if births_by_tick_sum == summary['total_births']:
            consistency_checks.append("✓ Tick totals match")
        else:
            consistency_checks.append("✗ Tick totals mismatch")
            
        if births_by_patch_sum == summary['total_births']:
            consistency_checks.append("✓ Patch totals match") 
        else:
            consistency_checks.append("✗ Patch totals mismatch")
            
        if len(birth_monitor.get_all_births()) == summary['total_births']:
            consistency_checks.append("✓ Birth indices count matches")
        else:
            consistency_checks.append("✗ Birth indices count mismatch")
        
        print("  Consistency checks:")
        for check in consistency_checks:
            print(f"    {check}")
        
        # Check that birth indices are unique
        all_births = birth_monitor.get_all_births()
        if len(all_births) == len(set(all_births)):
            print("    ✓ All birth indices are unique")
        else:
            print("    ✗ Duplicate birth indices found")
    
    model.cleanup()
    print("Data integrity test completed!")
    print()


if __name__ == "__main__":
    # Run the birth subscriber test first
    test_birth_has_subscribers()
    
    # Run the main birth monitor system test
    model, birth_monitor, birth_aware = test_birth_monitor_system()
    
    # Run data integrity test
    test_birth_data_integrity()
    
    # Optional: Interactive exploration
    print("\nComponents available for inspection:")
    print("  - model: ABMModel instance")
    print("  - birth_monitor: BirthMonitorTracker instance")  
    print("  - birth_aware: BirthAwareComponent instance")
    print("\nExample usage:")
    print("  birth_monitor.get_birth_summary()")
    print("  birth_monitor.get_all_births()")
    print("  model.event_bus.get_stats()")