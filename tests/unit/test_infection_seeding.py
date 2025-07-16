import importlib

import pytest

import laser_measles as lm
from laser_measles import MEASLES_MODULES


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_seed_single_patch(measles_module):
    """
    Test infection seeding for different model types.

    Args:
        measles_module (str): The module path to import as MeaslesModel.
    """
    MeaslesModel = importlib.import_module(measles_module)

    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))
    model.components = [
        MeaslesModel.components.InfectionSeedingProcess,
        MeaslesModel.components.InfectionProcess,
    ]  # NB: No disease progression included in the components
    model.run()
    if "E" in model.params.states:
        assert model.patches.states.E.sum() == 1
    else:
        assert model.patches.states.I.sum() == 1


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_seed_two_patch(measles_module):
    """Test the infection process in two patches."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=0))
    model.components = [
        MeaslesModel.components.InfectionSeedingProcess,
        MeaslesModel.components.InfectionProcess,
    ]  # NB: No disease progression included in the components
    model.run()
    if "E" in model.params.states:
        assert model.patches.states.E.sum() == 1
    else:
        assert model.patches.states.I.sum() == 1


if __name__ == "__main__":
    for module in MEASLES_MODULES:
        print(f"Testing {module}...")
        test_seed_single_patch(module)
        print(f"✓ {module} test passed")
