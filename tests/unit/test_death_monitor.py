#!/usr/bin/env python3
"""
Test script for the DeathMonitorTracker component.

This script creates a simple ABM model with vital dynamics and death monitoring
components to demonstrate and validate the death event system functionality.
Provides more comprehensive testing than the existing test_event_system.py.
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


def test_death_has_subscribers():
    """Test the has_subscribers functionality for death events."""
    print("Testing death event has_subscribers functionality")
    print("=" * 50)
    
    # Create a simple model for testing
    scenario = create_simple_scenario()
    params = ABMParams(num_ticks=1, verbose=False, show_progress=False, seed=12345)
    model = ABMModel(scenario, params, name="death_subscriber_test_model")
    
    # Create a test component with deaths enabled
    vital_params = VitalDynamicsParams(
        crude_death_rate=100.0,  # Enable deaths
        crude_birth_rate=0.0     # No births for cleaner testing
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
    print("\nTest 4: Test emit_event optimization for deaths")
    
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
    print("\ndeath has_subscribers test passed!")
    print()


def test_death_monitor_system():
    """Test the death monitoring system with a simple simulation."""
    print("Testing ABM Death Monitor System")
    print("=" * 60)
    
    # Create scenario and parameters
    scenario = create_simple_scenario()
    params = ABMParams(
        num_ticks=15,  # Longer simulation to ensure deaths occur
        verbose=True,
        show_progress=False,  # Disable progress bar for cleaner output
        seed=12345
    )
    
    print(f"Scenario: {len(scenario)} patches with {scenario['pop'].sum()} total population")
    print(f"Simulation: {params.num_ticks} ticks")
    print()
    
    # Create model
    model = ABMModel(scenario, params, name="death_test_model")
    
    # Setup components with vital dynamics and death monitoring
    # Use high death rate to ensure deaths occur during simulation
    vital_params = VitalDynamicsParams(
        crude_death_rate=150.0,  # High death rate for testing (150 per 1000 per year)
        crude_birth_rate=50.0    # Some births to balance population
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
    
    # Get death monitor results
    death_monitor = None
    death_aware = None
    
    for instance in model.instances:
        if isinstance(instance, DeathMonitorTracker):
            death_monitor = instance
        elif isinstance(instance, DeathAwareComponent):
            death_aware = instance
    
    # Validate death monitor results
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
        
        # Additional validation
        recent_deaths = death_monitor.get_recent_deaths()
        all_deaths = death_monitor.get_all_deaths()
        
        print("Death Data Validation:")
        print(f"  Recent deaths count: {len(recent_deaths)}")
        print(f"  Total unique deaths: {len(all_deaths)}")
        
        # Check that all recent deaths are in the all deaths set
        if recent_deaths:
            recent_in_all = all(idx in all_deaths for idx in recent_deaths)
            print(f"  Recent deaths in total set: {'✓' if recent_in_all else '✗'}")
        
        # Validate death indices are reasonable
        if all_deaths:
            min_idx = min(all_deaths)
            max_idx = max(all_deaths)
            print(f"  Death index range: {min_idx} to {max_idx}")
            
        # Validate disease state tracking
        if summary['deaths_by_state']:
            total_state_deaths = sum(summary['deaths_by_state'].values())
            print(f"  Deaths by state total: {total_state_deaths}")
            if total_state_deaths == summary['total_deaths']:
                print("  ✓ State death counts consistent")
            else:
                print("  ✗ State death counts inconsistent")
            
        print()
    
    if death_aware:
        print("Death Aware Component:")
        print(f"  Death reactions: {death_aware.get_reaction_count()}")
        print()
    
    # Verify the death monitoring system worked
    print("Death Monitor System Verification:")
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
        
        # Check if deaths occurred in multiple ticks
        deaths_by_tick = death_monitor.get_death_summary()['deaths_by_tick']
        if len(deaths_by_tick) > 1:
            print("  ✓ Deaths occurred across multiple ticks")
        else:
            print("  ! Deaths occurred in limited ticks (may be expected)")
            
        # Check if deaths occurred in multiple patches
        deaths_by_patch = death_monitor.get_death_summary()['deaths_by_patch']
        if len(deaths_by_patch) > 1:
            print("  ✓ Deaths occurred in multiple patches")
        else:
            print("  ! Deaths occurred in limited patches (may be expected)")
            
        # Check disease state distribution
        deaths_by_state = death_monitor.get_death_summary()['deaths_by_state']
        if deaths_by_state and len(deaths_by_state) >= 1:
            print("  ✓ Disease state tracking functional")
        else:
            print("  ! Limited disease state tracking")
            
    else:
        print("  ✗ No death events were processed")
    
    if death_aware and death_aware.get_reaction_count() > 0:
        print("  ✓ Components reacted to death events")
        
        # Validate reaction count matches death events
        if (death_monitor and 
            death_aware.get_reaction_count() == death_monitor.get_death_summary()['num_death_events']):
            print("  ✓ Reaction count matches death events")
        else:
            print("  ! Reaction count differs from death events")
    else:
        print("  ✗ No components reacted to death events")
    
    # Validate population dynamics
    if death_monitor:
        # Get population before and after
        initial_pop = scenario['pop'].sum()
        final_states = model.patches.states.sum()
        final_pop = final_states.sum() if hasattr(final_states, 'sum') else final_states
        
        print(f"\nPopulation Dynamics:")
        print(f"  Initial population: {initial_pop}")
        print(f"  Final population: {final_pop}")
        print(f"  Net change: {final_pop - initial_pop}")
        print(f"  Total deaths tracked: {death_monitor.total_deaths}")
    
    print()
    print("Death monitor test completed successfully!")
    
    # Cleanup
    model.cleanup()
    
    return None


def test_death_data_integrity():
    """Test the integrity of death data tracking."""
    print("Testing Death Data Integrity")
    print("=" * 40)
    
    # Create a smaller, controlled scenario
    scenario = pl.DataFrame({
        'pop': [500, 300],  # Smaller population for precise tracking
        'lat': [40.7, 40.8],
        'lon': [-74.0, -74.1], 
        'id': ['patch_1', 'patch_2']
    })
    
    params = ABMParams(
        num_ticks=8,  # Short simulation
        verbose=False,
        show_progress=False,
        seed=42  # Different seed for variety
    )
    
    model = ABMModel(scenario, params, name="death_integrity_test")
    
    # Configure components
    vital_params = VitalDynamicsParams(
        crude_death_rate=300.0,  # Very high rate to ensure deaths
        crude_birth_rate=0.0     # No births to simplify tracking
    )
    
    death_params = DeathMonitorParams(
        track_death_locations=True,
        track_death_states=True,
        verbose_deaths=False  # Reduce output for cleaner test
    )
    
    class TestVitalDynamics(VitalDynamicsProcess):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=vital_params)
    
    class TestDeathMonitor(DeathMonitorTracker):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=death_params)
    
    model.components = [TestVitalDynamics, TestDeathMonitor]
    
    print("Running integrity test simulation...")
    model.run()
    
    # Get the death monitor
    death_monitor = None
    for instance in model.instances:
        if isinstance(instance, DeathMonitorTracker):
            death_monitor = instance
            break
    
    if death_monitor:
        summary = death_monitor.get_death_summary()
        
        print("Data Integrity Checks:")
        print(f"  Total deaths: {summary['total_deaths']}")
        print(f"  Death events: {summary['num_death_events']}")
        
        # Check consistency between different metrics
        deaths_by_tick_sum = sum(summary['deaths_by_tick'].values())
        deaths_by_patch_sum = sum(summary['deaths_by_patch'].values())
        deaths_by_state_sum = sum(summary['deaths_by_state'].values())
        
        print(f"  Sum of deaths by tick: {deaths_by_tick_sum}")
        print(f"  Sum of deaths by patch: {deaths_by_patch_sum}")
        print(f"  Sum of deaths by state: {deaths_by_state_sum}")
        
        # Validate consistency
        consistency_checks = []
        if deaths_by_tick_sum == summary['total_deaths']:
            consistency_checks.append("✓ Tick totals match")
        else:
            consistency_checks.append("✗ Tick totals mismatch")
            
        if deaths_by_patch_sum == summary['total_deaths']:
            consistency_checks.append("✓ Patch totals match") 
        else:
            consistency_checks.append("✗ Patch totals mismatch")
            
        if deaths_by_state_sum == summary['total_deaths']:
            consistency_checks.append("✓ State totals match")
        else:
            consistency_checks.append("✗ State totals mismatch")
            
        if len(death_monitor.get_all_deaths()) == summary['total_deaths']:
            consistency_checks.append("✓ Death indices count matches")
        else:
            consistency_checks.append("✗ Death indices count mismatch")
        
        print("  Consistency checks:")
        for check in consistency_checks:
            print(f"    {check}")
        
        # Check that death indices are unique
        all_deaths = death_monitor.get_all_deaths()
        if len(all_deaths) == len(set(all_deaths)):
            print("    ✓ All death indices are unique")
        else:
            print("    ✗ Duplicate death indices found")
        
        # Check that death indices are within valid range (initial population)
        initial_pop = scenario['pop'].sum()
        if all_deaths and all(0 <= idx < initial_pop for idx in all_deaths):
            print("    ✓ All death indices within valid range")
        else:
            print("    ✗ Some death indices outside valid range")
            
        # Population decrease validation
        final_states = model.patches.states.sum()
        final_pop = final_states.sum() if hasattr(final_states, 'sum') else final_states
        expected_final = initial_pop - summary['total_deaths']
        
        print(f"  Population validation:")
        print(f"    Initial: {initial_pop}, Final: {final_pop}, Expected: {expected_final}")
        if final_pop == expected_final:
            print("    ✓ Population decrease matches death count")
        else:
            print(f"    ✗ Population decrease mismatch (diff: {final_pop - expected_final})")
    
    model.cleanup()
    print("Data integrity test completed!")
    print()


def test_death_with_disease_states():
    """Test death monitoring with disease state tracking (simplified test)."""
    print("Testing Death Monitor with Disease States")
    print("=" * 50)
    
    # Create scenario
    scenario = pl.DataFrame({
        'pop': [400, 600],  # Smaller population
        'lat': [40.7, 40.8],
        'lon': [-74.0, -74.1], 
        'id': ['patch_1', 'patch_2']
    })
    
    params = ABMParams(
        num_ticks=8,
        verbose=False,
        show_progress=False,
        seed=123
    )
    
    model = ABMModel(scenario, params, name="death_disease_test")
    
    # Configure components - just vital dynamics and death monitoring
    # This will test that disease state tracking works even with basic states
    vital_params = VitalDynamicsParams(
        crude_death_rate=250.0,  # High death rate
        crude_birth_rate=0.0     # No births
    )
    
    death_params = DeathMonitorParams(
        track_death_locations=True,
        track_death_states=True,
        verbose_deaths=False
    )
    
    # Create configured components
    class TestVitalDynamics(VitalDynamicsProcess):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=vital_params)
    
    class TestDeathMonitor(DeathMonitorTracker):
        def __init__(self, model, verbose=False):
            super().__init__(model, verbose=verbose, params=death_params)
    
    model.components = [
        TestVitalDynamics,
        TestDeathMonitor
    ]
    
    print("Running disease state test simulation...")
    model.run()
    
    # Get the death monitor
    death_monitor = None
    for instance in model.instances:
        if isinstance(instance, DeathMonitorTracker):
            death_monitor = instance
            break
    
    if death_monitor:
        summary = death_monitor.get_death_summary()
        
        print("Disease State Death Analysis:")
        print(f"  Total deaths: {summary['total_deaths']}")
        print(f"  Deaths by state: {summary['deaths_by_state']}")
        
        # Analyze state distribution
        if summary['deaths_by_state']:
            state_names = ['S', 'E', 'I', 'R']  # Typical SEIR states
            for state_idx, count in summary['deaths_by_state'].items():
                state_name = state_names[state_idx] if state_idx < len(state_names) else f"State_{state_idx}"
                print(f"    {state_name}: {count} deaths")
            
            # In a basic model, we expect most deaths to be from susceptible state (S)
            if 0 in summary['deaths_by_state']:  # State 0 = Susceptible
                print("  ✓ Disease state tracking functional (tracking susceptible deaths)")
            else:
                print("  ! Unexpected disease state distribution")
                
            # Validate state consistency
            total_state_deaths = sum(summary['deaths_by_state'].values())
            if total_state_deaths == summary['total_deaths']:
                print("  ✓ Disease state counts are consistent")
            else:
                print("  ✗ Disease state counts inconsistent")
        else:
            print("  ✗ No disease state data collected")
    
    model.cleanup()
    print("Disease state test completed!")
    print()


if __name__ == "__main__":
    # Run the death subscriber test first
    test_death_has_subscribers()
    
    # Run the main death monitor system test
    model, death_monitor, death_aware = test_death_monitor_system()
    
    # Run data integrity test
    test_death_data_integrity()
    
    # Run disease state test
    test_death_with_disease_states()
    
    # Optional: Interactive exploration
    print("\nComponents available for inspection:")
    print("  - model: ABMModel instance")
    print("  - death_monitor: DeathMonitorTracker instance")  
    print("  - death_aware: DeathAwareComponent instance")
    print("\nExample usage:")
    print("  death_monitor.get_death_summary()")
    print("  death_monitor.get_all_deaths()")
    print("  model.event_bus.get_stats()")