import importlib

import pytest

import laser_measles as lm
from laser_measles import MEASLES_MODULES

VERBOSE = False
SEED = 42


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_constant_pop_single_patch(measles_module):
    """Test the constant population scenario."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=50, verbose=VERBOSE, seed=SEED))
    model.components = [MeaslesModel.components.ConstantPopProcess]
    model.run()
    assert model.patches.states[:-1, :].sum() == scenario["pop"].sum()
    component = model.get_component(MeaslesModel.components.ConstantPopProcess)[0]
    assert component.mu_death == component.lambda_birth


@pytest.mark.slow
def test_ABM_pop_agreement():
    scenario = lm.scenarios.synthetic.two_patch_scenario()
    model = lm.abm.Model(scenario, lm.abm.Params(num_ticks=50, verbose=VERBOSE, seed=SEED))
    model.components = [lm.abm.components.ConstantPopProcess]
    model.run()
    # Assert population between patches and people are in agreement
    assert model.patches.states.sum() == len(model.people)


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_constant_pop_two_patch(measles_module):
    """Test the constant population scenario."""
    MeaslesModel = importlib.import_module(measles_module)
    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=50, verbose=VERBOSE, seed=SEED))
    model.components = [MeaslesModel.components.ConstantPopProcess]
    model.run()
    assert model.patches.states[:-1, :].sum() == scenario["pop"].sum()


if __name__ == "__main__":
    for module in MEASLES_MODULES:
        print(f"Testing {module}...")
        test_constant_pop_single_patch(module)
        print(f"✓ {module} constant pop test passed")
        test_constant_pop_two_patch(module)
        print(f"✓ {module} constant pop test passed")
