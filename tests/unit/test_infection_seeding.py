import importlib

import pytest

import laser_measles as lm

MEASLES_MODULES = ["laser_measles.biweekly", "laser_measles.compartmental", "laser_measles.abm"]


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_seed_single_patch(measles_module):
    """
    Test infection seeding for different model types.

    Args:
        measles_module (str): The module path to import as MeaslesModel.
    """
    MeaslesModel = importlib.import_module(measles_module)

    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=25), name="test_seed_single_patch")
    model.components = [MeaslesModel.components.InfectionSeedingProcess]  # NB: No disease progression included in the components
    model.run()
    assert model.patches.states.I.sum() == 1


if __name__ == "__main__":
    for module in MEASLES_MODULES:
        print(f"Testing {module}...")
        test_seed_single_patch(module)
        print(f"✓ {module} test passed")
