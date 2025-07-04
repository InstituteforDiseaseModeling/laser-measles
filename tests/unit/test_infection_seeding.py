import pytest
import importlib
import laser_measles as lm

@pytest.mark.parametrize("measles_module", [
    "laser_measles.biweekly",
    "laser_measles.compartmental"
])
def test_seed_single_patch(measles_module):
    """
    Test infection seeding for different model types.

    Args:
        measles_module (str): The module path to import as MeaslesModel.
    """
    MeaslesModel = importlib.import_module(measles_module)

    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.two_cluster_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=25), name="test_seed_single_patch")
    model.components = [MeaslesModel.components.InfectionSeedingProcess]
    model.run()
    assert model.patches.states.I.sum() == 1

if __name__ == "__main__":
    for module in ["laser_measles.biweekly", "laser_measles.compartmental"]:
        print(f"Testing {module}...")
        test_seed_single_patch(module)
        print(f"✓ {module} test passed")