import importlib

import numpy as np
import pytest

import laser.measles as lm
from laser.measles import MEASLES_MODULES

VERBOSE = False
SEED = 42


def expected_growth(model, module) -> np.ndarray:
    """Expected growth of the population."""
    component = model.get_component(module.components.VitalDynamicsProcess)[0]
    rate = component.lambda_birth - component.mu_death  # calculated per tick
    N = model.scenario["pop"].to_numpy() * np.exp(rate * model.params.num_ticks)
    return np.array(N)


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_vital_dynamics_single_patch(measles_module):
    """Test the vital dynamics in a single patch."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=365, verbose=VERBOSE, seed=SEED))
    model.components = [MeaslesModel.components.VitalDynamicsProcess]
    model.run()
    expected = expected_growth(model, MeaslesModel)
    assert model.patches.states.sum(axis=0) > model.scenario["pop"].sum()
    assert np.abs(model.patches.states.sum(axis=0) - expected) / expected < 0.10


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_vital_dynamics_two_patch(measles_module):
    """Test the vital dynamics in two patches."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=365, verbose=VERBOSE, seed=SEED))
    model.components = [MeaslesModel.components.VitalDynamicsProcess]
    model.run()
    expected = expected_growth(model, MeaslesModel)
    assert np.sum(model.patches.states) > model.scenario["pop"].sum()
    assert np.all(np.abs(model.patches.states.sum(axis=0) - expected) / expected < 0.10)


@pytest.mark.parametrize("measles_module", ["laser.measles.abm"])
def test_infection_with_vital_dynamics_no_underflow(measles_module):
    """Regression test: InfectionProcess(beta>0) must not underflow patches.states.S.

    When an epidemic burns through all susceptibles in a single tick, the ABM's
    InfectionProcess can decrement patches.states.S (uint32) below zero, wrapping it
    to ~4 294 967 273. VitalDynamicsProcess then computes deaths from this inflated
    count and crashes with:
        ValueError: Cannot take a larger sample than population when replace=False

    This is the community-transmission complement to test_importation_with_vital_dynamics
    (which covers the same underflow triggered by ImportationPressureProcess at beta=0).

    The fix: InfectionProcess must clamp the patch-state decrement so that
    patches.states.S never goes below 0.
    """
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=100, verbose=VERBOSE, seed=SEED))
    model.components = [
        MeaslesModel.components.VitalDynamicsProcess,
        lm.create_component(
            MeaslesModel.components.InfectionSeedingProcess,
            MeaslesModel.components.InfectionSeedingParams(num_infections=10),
        ),
        lm.create_component(
            MeaslesModel.components.InfectionProcess,
            MeaslesModel.components.InfectionParams(beta=20.0),
        ),
    ]

    # Currently crashes with ValueError before reaching assertions.
    # After the fix this should complete and both assertions should hold.
    model.run()

    # S values must be physically meaningful (no uint32 wrap-around)
    assert np.all(model.patches.states.S < 1_000_000), f"patches.states.S looks like uint32 underflow: {model.patches.states.S}"

    # Population must be conserved: patch state counts == active agent count
    active_count = model.people.active[: model.people.count].sum()
    state_total = np.sum(model.patches.states)
    assert state_total == active_count, f"State total {state_total} != active agent count {active_count}"


if __name__ == "__main__":
    for module in MEASLES_MODULES:
        print(f"Testing {module}...")
        test_vital_dynamics_single_patch(module)
        print(f"✓ {module} single patch test passed")

        test_vital_dynamics_two_patch(module)
        print(f"✓ {module} two patch test passed")

    print("Testing ABM-only regression...")
    test_infection_with_vital_dynamics_no_underflow("laser.measles.abm")
    print("✓ ABM infection+vital_dynamics underflow test passed")
