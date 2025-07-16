import importlib

import numpy as np
import pytest

import laser_measles as lm
from laser_measles import MEASLES_MODULES

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


if __name__ == "__main__":
    for module in MEASLES_MODULES:
        print(f"Testing {module}...")
        test_vital_dynamics_single_patch(module)
        print(f"✓ {module} single patch test passed")

        test_vital_dynamics_two_patch(module)
        print(f"✓ {module} two patch test passed")
