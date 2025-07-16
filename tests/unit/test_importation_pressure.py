import importlib

import numpy as np
import pytest

import laser_measles as lm
from laser_measles import MEASLES_MODULES

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


if __name__ == "__main__":
    pytest.main([__file__ + "::test_importation_pressure_two_patch", "-v", "-s"])
