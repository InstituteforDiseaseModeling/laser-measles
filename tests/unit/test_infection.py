import importlib

import numpy as np
import pytest

import laser_measles as lm
from laser_measles import MEASLES_MODULES

VERBOSE = False
SEED = 42


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_infection_single_patch(measles_module):
    """Test the infection process in a single patch."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=50, verbose=VERBOSE, seed=SEED))
    model.components = [
        lm.create_component(
            MeaslesModel.components.InfectionSeedingProcess, MeaslesModel.components.InfectionSeedingParams(num_infections=10)
        ),
        MeaslesModel.components.InfectionProcess,
    ]
    model.run()
    if VERBOSE:
        print(
            f"Final fraction recovered: {100 * model.patches.states.R.sum() / scenario['pop'].sum():.2f}% (N={model.patches.states.R.sum()})"
        )
    assert model.patches.states.R.sum() > 10


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_infection_two_patch(measles_module):
    """Test the infection process in two patches."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=25, verbose=VERBOSE, seed=SEED))
    model.components = [
        lm.create_component(
            MeaslesModel.components.InfectionSeedingProcess, MeaslesModel.components.InfectionSeedingParams(num_infections=10)
        ),
        MeaslesModel.components.InfectionProcess,
    ]
    model.run()
    if VERBOSE:
        print(
            f"Final fraction recovered: {100 * model.patches.states.R.sum() / scenario['pop'].sum():.2f}% (N={model.patches.states.R.sum()})"
        )
    assert np.all(model.patches.states.R >= 0)
    assert model.patches.states.R.sum() > 10


if __name__ == "__main__":
    for module in MEASLES_MODULES:
        print(f"Testing {module}...")
        test_infection_single_patch(module)
        print(f"✓ {module} single patch test passed")

        test_infection_two_patch(module)
        print(f"✓ {module} two patch test passed")
