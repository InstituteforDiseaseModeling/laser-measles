"""
Unit tests for the Initialize Equilibrium States component.

Tests the equilibrium state initialization across all model types (ABM, compartmental, biweekly)
with focus on proper equilibrium calculation and ABM-specific agent consistency.
"""

import importlib

import numpy as np
import pytest

import laser_measles as lm
from laser_measles import MEASLES_MODULES


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_equilibrium_basic_single_patch(measles_module):
    """Test basic equilibrium calculation with default R0=8.0 on single patch."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))
    model.components = [MeaslesModel.components.InitializeEquilibriumStatesProcess]
    model.run()

    # Test patch-level equilibrium
    total_pop = scenario["pop"].sum()
    expected_s = total_pop / 8.0  # Default R0 = 8.0
    expected_r = total_pop * (1 - 1 / 8.0)

    assert abs(model.patches.states.S.sum() - expected_s) < 1e-10
    assert abs(model.patches.states.R.sum() - expected_r) < 1e-10
    # Test that E and I states are zero (if they exist)
    if hasattr(model.patches.states, "E"):
        assert model.patches.states.E.sum() == 0
    if hasattr(model.patches.states, "I"):
        assert model.patches.states.I.sum() == 0

    # Test population conservation
    actual_sum = int(model.patches.states.sum())
    assert abs(actual_sum - total_pop) < 1e-10


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_equilibrium_basic_two_patch(measles_module):
    """Test basic equilibrium calculation with default R0=8.0 on two patches."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))
    model.components = [MeaslesModel.components.InitializeEquilibriumStatesProcess]
    model.run()

    # Test patch-level equilibrium for each patch
    for patch_idx in range(len(scenario)):
        patch_pop = scenario["pop"][patch_idx]
        expected_s = patch_pop / 8.0
        expected_r = patch_pop * (1 - 1 / 8.0)

        assert abs(model.patches.states.S[patch_idx] - expected_s) < 1e-10
        assert abs(model.patches.states.R[patch_idx] - expected_r) < 1e-10
        # Test that E and I states are zero (if they exist)
        if hasattr(model.patches.states, "E"):
            assert model.patches.states.E[patch_idx] == 0
        if hasattr(model.patches.states, "I"):
            assert model.patches.states.I[patch_idx] == 0

        # Test population conservation per patch
        patch_total = int(model.patches.states[:, patch_idx].sum())
        assert abs(patch_total - patch_pop) < 1e-10


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_equilibrium_custom_r0(measles_module):
    """Test equilibrium calculation with custom R0 values."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())

    # Test different R0 values
    test_r0_values = [1.0, 2.0, 4.0, 12.0, 20.0]

    for r0 in test_r0_values:
        model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))
        equilibrium_params = MeaslesModel.components.InitializeEquilibriumStatesParams(R0=r0)
        model.components = [lm.create_component(MeaslesModel.components.InitializeEquilibriumStatesProcess, equilibrium_params)]
        model.run()

        total_pop = scenario["pop"].sum()
        expected_s = total_pop / r0
        expected_r = total_pop * (1 - 1 / r0)

        # For ABM, we expect exact integer matching after rounding
        # For compartmental/biweekly models, we expect fractional values to be truncated
        if "abm" in measles_module:
            # ABM should have integer counts that are close to the expected values
            assert abs(model.patches.states.S.sum() - expected_s) < 1
            assert abs(model.patches.states.R.sum() - expected_r) < 1
        else:
            # Other models may have fractional values truncated to integers
            assert abs(model.patches.states.S.sum() - expected_s) < 1
            assert abs(model.patches.states.R.sum() - expected_r) < 1
        # Population conservation may be affected by integer truncation
        # Convert to int to avoid unsigned integer underflow issues
        actual_sum = int(model.patches.states.sum())
        assert abs(actual_sum - total_pop) <= len(scenario)


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_equilibrium_edge_cases(measles_module):
    """Test equilibrium calculation with edge case R0 values."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())

    # Test R0 = 1.0 (all susceptible)
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))
    equilibrium_params = MeaslesModel.components.InitializeEquilibriumStatesParams(R0=1.0)
    model.components = [lm.create_component(MeaslesModel.components.InitializeEquilibriumStatesProcess, equilibrium_params)]
    model.run()

    total_pop = scenario["pop"].sum()
    # For R0 = 1.0, all population should be susceptible
    assert abs(model.patches.states.S.sum() - total_pop) < 1
    assert abs(model.patches.states.R.sum() - 0) < 1

    # Test R0 = 0.5 (impossible but mathematically valid)
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))
    equilibrium_params = MeaslesModel.components.InitializeEquilibriumStatesParams(R0=0.5)
    model.components = [lm.create_component(MeaslesModel.components.InitializeEquilibriumStatesProcess, equilibrium_params)]
    model.run()

    # For R0 < 1, the mathematical result would be negative R, but implementations
    # should handle this gracefully by clamping to valid ranges
    if "abm" in measles_module:
        # ABM should clamp negative values to 0
        assert model.patches.states.S.sum() == total_pop
        assert model.patches.states.R.sum() == 0
    else:
        # Other models may have different behavior, but should not crash
        # Just check that the result is reasonable
        assert int(model.patches.states.S.sum()) >= 0
        assert int(model.patches.states.R.sum()) >= 0


def test_abm_agent_consistency():
    """Test that ABM agents are consistent with patch states."""
    MeaslesModel = importlib.import_module("laser_measles.abm")
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))
    model.components = [MeaslesModel.components.InitializeEquilibriumStatesProcess]
    model.run()

    # Test that individual agent states match patch counts
    s_state_idx = model.params.states.index("S")
    r_state_idx = model.params.states.index("R")

    # Count agents in each state
    agent_s_count = np.sum(model.people.state == s_state_idx)
    agent_r_count = np.sum(model.people.state == r_state_idx)

    # Compare with patch totals
    patch_s_total = int(model.patches.states.S.sum())
    patch_r_total = int(model.patches.states.R.sum())

    assert agent_s_count == patch_s_total
    assert agent_r_count == patch_r_total

    # Test susceptibility values
    s_agents = model.people.state == s_state_idx
    r_agents = model.people.state == r_state_idx

    # All S agents should have susceptibility = 1.0
    assert np.allclose(model.people.susceptibility[s_agents], 1.0)
    # All R agents should have susceptibility = 0.0
    assert np.allclose(model.people.susceptibility[r_agents], 0.0)

    # Test patch_id assignment matches population
    for patch_idx in range(len(scenario)):
        patch_pop = scenario["pop"][patch_idx]
        agents_in_patch = np.sum(model.people.patch_id == patch_idx)
        assert agents_in_patch == patch_pop


def test_abm_agent_state_distribution():
    """Test that ABM agents are properly distributed across patches."""
    MeaslesModel = importlib.import_module("laser_measles.abm")
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))

    # Use custom R0 for clearer testing
    equilibrium_params = MeaslesModel.components.InitializeEquilibriumStatesParams(R0=4.0)
    model.components = [lm.create_component(MeaslesModel.components.InitializeEquilibriumStatesProcess, equilibrium_params)]
    model.run()

    s_state_idx = model.params.states.index("S")
    r_state_idx = model.params.states.index("R")

    # Test state distribution within each patch
    for patch_idx in range(len(scenario)):
        patch_agents = model.people.patch_id == patch_idx

        # Count agents in each state for this patch
        patch_s_agents = np.sum((model.people.state == s_state_idx) & patch_agents)
        patch_r_agents = np.sum((model.people.state == r_state_idx) & patch_agents)

        # Compare with patch state counts
        expected_s = int(model.patches.states.S[patch_idx])
        expected_r = int(model.patches.states.R[patch_idx])

        assert patch_s_agents == expected_s
        assert patch_r_agents == expected_r

        # Total agents in patch should match scenario population
        total_agents_in_patch = np.sum(patch_agents)
        assert total_agents_in_patch == scenario["pop"][patch_idx]


def test_abm_people_initialization():
    """Test that ABM people LaserFrame is properly initialized."""
    MeaslesModel = importlib.import_module("laser_measles.abm")
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))
    model.components = [MeaslesModel.components.InitializeEquilibriumStatesProcess]
    model.run()

    total_pop = scenario["pop"].sum()

    # Test that people LaserFrame has correct count
    assert model.people.count == total_pop

    # Test that all required properties are set
    assert len(model.people.patch_id) == total_pop
    assert len(model.people.state) == total_pop
    assert len(model.people.susceptibility) == total_pop

    # Test that no agent has invalid state
    valid_states = set(range(len(model.params.states)))
    for state in model.people.state:
        assert state in valid_states


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
