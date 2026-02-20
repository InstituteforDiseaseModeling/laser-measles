import importlib

import numpy as np
import pytest

import laser.measles as lm
from laser.measles import MEASLES_MODULES

VERBOSE = False
SEED = 42


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_importation_pressure_single_patch(measles_module):
    """Test the infection process in a single patch."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=100, verbose=VERBOSE, seed=SEED))
    model.components = [
        MeaslesModel.components.ImportationPressureProcess,
        lm.create_component(MeaslesModel.components.InfectionProcess, MeaslesModel.components.InfectionParams(beta=1.0)),
    ]
    model.run()
    if VERBOSE:
        print(
            f"Final fraction recovered: {100 * model.patches.states.R.sum() / scenario['pop'].sum():.2f}% (N={model.patches.states.R.sum()})"
        )
    assert model.patches.states.R.sum() > 1
    assert np.sum(model.patches.states) == np.sum(model.scenario["pop"].to_numpy())


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_importation_pressure_two_patch(measles_module):
    """Test the infection process in two patches."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=100, verbose=VERBOSE, seed=SEED))
    model.components = [
        MeaslesModel.components.ImportationPressureProcess,
        lm.create_component(MeaslesModel.components.InfectionProcess, MeaslesModel.components.InfectionParams(beta=0.0)),
    ]
    model.run()
    if VERBOSE:
        print(
            f"Final fraction recovered: {100 * model.patches.states.R.sum() / scenario['pop'].sum():.2f}% (N={model.patches.states.R.sum()})"
        )
    assert np.all(model.patches.states.R >= 1)
    assert model.patches.states.R.sum() > 1
    assert np.all(np.equal(model.patches.states.sum(axis=0), scenario["pop"].to_numpy()))


@pytest.mark.parametrize("measles_module", ["laser.measles.abm"])
def test_importation_with_vital_dynamics(measles_module):
    """Regression test: ImportationPressureProcess should not infect inactive (unborn) agents.

    Without filtering by `active`, phantom agents beyond people.count get infected,
    causing S[0] to underflow (uint32 wrap) and VitalDynamicsProcess to compute
    massive birth counts that exhaust array capacity around tick 32-50.

    Uses beta=0.0 to disable community transmission and isolate the importation fix.
    """
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=50, verbose=VERBOSE, seed=SEED))
    model.components = [
        MeaslesModel.components.VitalDynamicsProcess,
        MeaslesModel.components.ImportationPressureProcess,
        lm.create_component(MeaslesModel.components.InfectionProcess, MeaslesModel.components.InfectionParams(beta=0.0)),
    ]
    model.run()

    # S values should remain reasonable (no uint32 underflow)
    assert np.all(model.patches.states.S < 1_000_000), f"S values look like uint32 underflow: {model.patches.states.S}"

    # Population conservation: state counts should equal active agent count
    active_count = model.people.active[: model.people.count].sum()
    state_total = np.sum(model.patches.states)
    assert state_total == active_count, f"State total {state_total} != active count {active_count}"


if __name__ == "__main__":
    pytest.main([__file__ + "::test_importation_pressure_two_patch", "-v", "-s"])
