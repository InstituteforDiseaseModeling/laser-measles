import numpy as np
import pytest

from laser_measles.abm.components import WPPVitalDynamicsProcess, PopulationTracker

from laser_measles.abm.model import ABMModel
from laser_measles.abm.model import ABMParams
from laser_measles.scenarios.synthetic import two_patch_scenario


@pytest.fixture
def WPPModelZeroTicks():
    scenario = two_patch_scenario(population=1000)
    params = ABMParams(num_ticks=0)
    model = ABMModel(scenario, params)
    model.components = [WPPVitalDynamicsProcess]
    model.run()
    return model

@pytest.fixture
def WPPModel():
    scenario = two_patch_scenario(population=100_000)
    params = ABMParams(num_ticks=5*365, start_time='2000-06', seed=12)
    model = ABMModel(scenario, params)
    model.components = [WPPVitalDynamicsProcess]
    model.run()
    return model

def test_initial_node_ids(WPPModelZeroTicks):
    # Check that the number of people in each patch is correct
    for i, row in enumerate(WPPModelZeroTicks.scenario.iter_rows(named=True)):
        assert np.sum(np.logical_and(WPPModelZeroTicks.people.patch_id == i, WPPModelZeroTicks.people.active)) == row["pop"]

@pytest.mark.slow
def test_pop_agreement(WPPModel):
    # Assert population between patches and people are in agreement
    assert WPPModel.patches.states.sum() == WPPModel.people.active.sum()

@pytest.mark.slow
def test_wpp_vital_dynamics(WPPModel):
    vd = WPPModel.get_component(WPPVitalDynamicsProcess)[0]
    initial_pyramid = vd.pyramid_spline(WPPModel.start_time.year)
    final_pyramid = vd.pyramid_spline(WPPModel.start_time.year + WPPModel.params.num_ticks // 365)
    wpp_growth_rate = (final_pyramid.sum() - initial_pyramid.sum()) / initial_pyramid.sum()
    model_growth_rate = (WPPModel.people.active.sum()  - WPPModel.scenario['pop'].sum()) / WPPModel.scenario['pop'].sum()
    assert np.isclose(model_growth_rate, wpp_growth_rate, atol=2e-2)

if __name__ == "__main__":
    pytest.main([__file__ + "::test_wpp_vital_dynamics", "-v", "-s"])
