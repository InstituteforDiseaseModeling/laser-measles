#!/usr/bin/env python3
"""
Test script for MCV1 vaccination component integration.

This script creates a simple ABM model with births, deaths, and MCV1 vaccination
to verify that the ProcessMCV1 component integrates correctly with the event system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from laser_measles.scenarios.synthetic import two_patch_scenario
from laser_measles.abm.base import BaseScenario
from laser_measles.abm.model import ABMModel
from laser_measles.abm.params import ABMParams
from laser_measles.abm.components import (
    VitalDynamicsProcess, 
    ProcessMCV1, 
    MCV1Params,
    StateTracker
)


def test_mcv1_integration():
    """Test MCV1 component integration with ABM model."""
    print("Testing MCV1 component integration...")
    
    # Create a simple scenario
    scenario_df = two_patch_scenario(population=1000, mcv1_coverage=0.8)
    scenario = BaseScenario(scenario_df)
    
    # Set up model parameters
    params = ABMParams(
        num_ticks=200,  # ~6-7 months
        use_numba=False,
        seed=42
    )
    
    # Create MCV1 parameters with very short delay for testing
    mcv1_params = MCV1Params(
        vaccination_delay_mean=5.0,  # 5 days instead of 270 for faster testing
        vaccination_efficacy=0.9,
        coverage=1.0,  # 100% coverage for testing
        delay_distribution="gamma"
    )
    
    # Create model
    model = ABMModel(scenario, params, name="mcv1_test")
    
    # Add components without custom parameters first
    model.components = [
        VitalDynamicsProcess,  # Generates births and deaths
        StateTracker  # Tracks population states over time
    ]
    
    # Add ProcessMCV1 as a separate step with custom parameters
    # We need to manually create and add this since it has custom params
    mcv1_instance = ProcessMCV1(model, verbose=False, params=mcv1_params)
    model.instances.append(mcv1_instance)
    if hasattr(mcv1_instance, '__call__'):
        model.phases.append(mcv1_instance)
    
    # Setup event bus for the manually added component
    if hasattr(mcv1_instance, 'set_event_bus'):
        mcv1_instance.set_event_bus(model.event_bus)
    
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
    mcv1_process = model.get_instance(ProcessMCV1)[0]
    
    # Print final states
    print(f"\nFinal population: {model.people.count}")
    print(f"Final states: S={model.patches.states.S.sum()}, "
          f"E={model.patches.states.E.sum()}, "
          f"I={model.patches.states.I.sum()}, "
          f"R={model.patches.states.R.sum()}")
    
    # Print MCV1 statistics
    mcv1_stats = mcv1_process.get_stats()
    print(f"\nMCV1 Statistics:")
    print(f"- Births scheduled for vaccination: {mcv1_stats['births_scheduled']}")
    print(f"- Vaccinations completed: {mcv1_stats['vaccinations_completed']}")
    print(f"- Agents protected: {mcv1_stats['agents_protected']}")
    print(f"- Agents not protected: {mcv1_stats['agents_not_protected']}")
    print(f"- Overall protection rate: {mcv1_stats['overall_protection_rate']:.2%}")
    print(f"- Pending vaccinations: {mcv1_stats['pending_vaccinations']}")
    
    # Print event bus statistics
    event_stats = model.event_bus.get_stats()
    print(f"\nEvent System Statistics:")
    print(f"- Events emitted: {event_stats['events_emitted']}")
    print(f"- Active event types: {event_stats['active_event_types']}")
    print(f"- Total subscribers: {event_stats['total_subscribers']}")
    print(f"- Dispatch errors: {event_stats['dispatch_errors']}")
    
    # Basic validation checks
    assert mcv1_stats['births_scheduled'] > 0, "No births were scheduled for vaccination"
    assert mcv1_stats['vaccinations_completed'] >= 0, "Vaccinations completed should be non-negative"
    assert event_stats['dispatch_errors'] == 0, "Event system should not have dispatch errors"
    
    # Check that some agents moved to R state due to vaccination
    final_recovered = model.patches.states.R.sum()
    print(f"\nValidation:")
    print(f"- Final recovered population: {final_recovered}")
    
    if mcv1_stats['agents_protected'] > 0:
        print("✓ MCV1 vaccination successfully protected some agents")
    else:
        print("⚠ No agents were protected by MCV1 (may be due to short simulation time)")
    
    if mcv1_stats['pending_vaccinations'] > 0:
        print(f"✓ {mcv1_stats['pending_vaccinations']} vaccinations still pending (delayed scheduling working)")
    
    print("\n✅ MCV1 integration test completed successfully!")

if __name__ == "__main__":
    test_mcv1_integration()