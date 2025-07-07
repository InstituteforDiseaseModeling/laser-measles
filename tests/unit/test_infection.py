import importlib

import numpy as np
import pytest

import laser_measles as lm

MEASLES_MODULES = ["laser_measles.biweekly", "laser_measles.compartmental", "laser_measles.abm"]
VERBOSE = False
MEASLES_MODULES = MEASLES_MODULES[1:]
SEED = np.random.randint(0, 1000000)
print(f"Using seed: {SEED}")

@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_infection_single_patch(measles_module):
    """ Test the infection process in a single patch. """

    MeaslesModel = importlib.import_module(measles_module)

    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=50, verbose=VERBOSE, seed=SEED), name="test_infection_single_patch")
    model.components = [
        MeaslesModel.components.InfectionSeedingProcess,
        MeaslesModel.components.InfectionProcess]
    model.run()
    print(f"Final fraction recovered: {100*model.patches.states.R.sum() / scenario['pop'].sum():.2f}%")
    print(f"Final number recovered: {model.patches.states.R.sum()}")
    assert model.patches.states.R.sum() > 1

@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_infection_two_patch(measles_module):
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=25, verbose=VERBOSE, seed=SEED), name="test_infection_double_patch")
    model.components = [
        MeaslesModel.components.InfectionSeedingProcess,
        MeaslesModel.components.InfectionProcess]
    model.run()
    print(f"Final fraction recovered: {100*model.patches.states.R.sum() / scenario['pop'].sum():.2f}%")
    print(f"Final number recovered: {model.patches.states.R.sum()}")
    assert model.patches.states.R.sum() > 1

if __name__ == "__main__":
    for module in MEASLES_MODULES:
        print(f"Testing {module}...")
        test_infection_single_patch(module)
        print(f"✓ {module} single patch test passed")

        # test_infection_two_patch(module)
        # print(f"✓ {module} two patch test passed")


