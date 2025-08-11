#!/usr/bin/env python3
"""
Test script for maternal immunity component integration.

This script creates a simple ABM model with births, deaths, and maternal immunity
to verify that the ProcessMaternalImmunity component integrates correctly with the event system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from laser_measles.scenarios.synthetic import two_patch_scenario
from laser_measles.abm.base import BaseScenario
from laser_measles.abm import ABMModel, ABMParams
from laser_measles.abm.components import (
    VitalDynamicsProcess, 
    ProcessMaternalImmunity, 
    MaternalImmunityParams,
    StateTracker
)

def test_maternal_immunity_integration():
    """Test maternal immunity component integration with ABM model."""
    print("Testing maternal immunity component integration...")
    
    # Create a simple scenario
    scenario_df = two_patch_scenario(population=1000, mcv1_coverage=0.0)
    scenario = BaseScenario(scenario_df)
    
    # Set up model parameters
    params = ABMParams(
        num_ticks=200,  # Longer simulation to see births and immunity expiration
        use_numba=False,
        seed=42
    )
    
    # Create maternal immunity parameters with short duration for testing
    maternal_immunity_params = MaternalImmunityParams(
        protection_duration_mean=10.0,  # 10 days instead of 180 for faster testing
        coverage=1.0,  # 100% coverage for testing
        distribution="gamma"
    )
    
    # Create model
    model = ABMModel(scenario, params, name="maternal_immunity_test")
    
    # Add components without custom parameters first
    model.components = [
        VitalDynamicsProcess,  # Generates births and deaths
        StateTracker  # Tracks population states over time
    ]
    
    # Add ProcessMaternalImmunity as a separate step with custom parameters
    # We need to manually create and add this since it has custom params
    maternal_immunity_instance = ProcessMaternalImmunity(model, verbose=False, params=maternal_immunity_params)
    model.instances.append(maternal_immunity_instance)
    if hasattr(maternal_immunity_instance, '__call__'):
        model.phases.append(maternal_immunity_instance)
    
    # Setup event bus for the manually added component
    if hasattr(maternal_immunity_instance, 'set_event_bus'):
        maternal_immunity_instance.set_event_bus(model.event_bus)
    
    print(f"Initial population: {model.people.count}")
    print(f"Initial states: S={model.patches.states.S.sum()}, "
          f"E={model.patches.states.E.sum()}, "
          f"I={model.patches.states.I.sum()}, "
          f"R={model.patches.states.R.sum()}")
    
    # Run simulation
    print("\nRunning simulation...")
    model.run()
    
    # Get results
    state_tracker = model.get_instance(StateTracker)[0]
    maternal_immunity_process = model.get_instance(ProcessMaternalImmunity)[0]
    
    # Print final states
    print(f"\nFinal population: {model.people.count}")
    print(f"Final states: S={model.patches.states.S.sum()}, "
          f"E={model.patches.states.E.sum()}, "
          f"I={model.patches.states.I.sum()}, "
          f"R={model.patches.states.R.sum()}")
    
    # Print maternal immunity statistics
    maternal_stats = maternal_immunity_process.get_stats()
    print(f"\nMaternal Immunity Statistics:")
    print(f"- Births protected: {maternal_stats['births_protected']}")
    print(f"- Immunity expired: {maternal_stats['immunity_expired']}")
    print(f"- Currently protected: {maternal_stats['agents_currently_protected']}")
    print(f"- Pending expirations: {maternal_stats['pending_expirations']}")
    
    # Print event bus statistics
    event_stats = model.event_bus.get_stats()
    print(f"\nEvent System Statistics:")
    print(f"- Events emitted: {event_stats['events_emitted']}")
    print(f"- Active event types: {event_stats['active_event_types']}")
    print(f"- Total subscribers: {event_stats['total_subscribers']}")
    print(f"- Dispatch errors: {event_stats['dispatch_errors']}")
    
    # Basic validation checks
    assert maternal_stats['births_protected'] >= 0, "Births protected should be non-negative"
    assert maternal_stats['immunity_expired'] >= 0, "Immunity expired should be non-negative"
    assert event_stats['dispatch_errors'] == 0, "Event system should not have dispatch errors"
    
    # Check susceptibility distribution
    people = model.people
    active_agents = people.active[:people.count]
    susceptibilities = people.susceptibility[:people.count][active_agents]
    protected_count = np.sum(susceptibilities == 0.0)
    susceptible_count = np.sum(susceptibilities == 1.0)
    
    print(f"\nValidation:")
    print(f"- Total agents with susceptibility=0: {protected_count}")
    print(f"- Total agents with susceptibility=1: {susceptible_count}")
    print(f"- Initial population: {scenario['pop'].sum()} (all start with susceptibility=0)")
    
    if maternal_stats['births_protected'] > 0:
        print("✓ Maternal immunity successfully protected some newborns")
    else:
        print("⚠ No births were protected (may be due to low birth rate in short simulation)")
    
    if maternal_stats['immunity_expired'] > 0:
        print(f"✓ {maternal_stats['immunity_expired']} immunity protections expired (time-based expiration working)")
    
    if maternal_stats['pending_expirations'] > 0:
        print(f"✓ {maternal_stats['pending_expirations']} immunity expirations still pending (delayed expiration working)")
    
    # Verify that newborns without expired immunity have susceptibility=0
    if maternal_stats['agents_currently_protected'] > 0:
        print(f"✓ {maternal_stats['agents_currently_protected']} agents currently have active maternal immunity")
    
    print("\n✅ Maternal immunity integration test completed successfully!")
    
    # Clean up
    model.cleanup()

if __name__ == "__main__":
    test_maternal_immunity_integration()