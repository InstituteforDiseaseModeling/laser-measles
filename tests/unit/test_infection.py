import laser_measles as lm
import pytest
import importlib

@pytest.mark.parametrize("measles_module", [
    "laser_measles.biweekly",
    "laser_measles.compartmental"
])
def test_infection_single_patch(measles_module):
    MeaslesModel = importlib.import_module(measles_module)

    scenario = MeaslesModel.BaseScenario(lm.scenarios.synthetic.single_patch_scenario())
    model = MeaslesModel.Model(scenario, MeaslesModel.Params(num_ticks=25), name="test_infection_single_patch")
    model.components = [
        MeaslesModel.components.InfectionSeedingProcess,
        MeaslesModel.components.InfectionProcess]
    model.run()
    assert model.patches.states.R.sum() > 1

if __name__ == "__main__":
    for module in ["laser_measles.biweekly", "laser_measles.compartmental"]:
        print(f"Testing {module}...")
        test_infection_single_patch(module)
        print(f"✓ {module} test passed")
