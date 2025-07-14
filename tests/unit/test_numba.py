"""
Test numba acceleration
"""

import timeit

import pytest

import laser_measles as lm
from laser_measles.scenarios.synthetic import two_patch_scenario


@pytest.fixture
def test_scenario_df():
    return two_patch_scenario(population=100_000)


@pytest.fixture
def numba_model(test_scenario_df):
    """Fixture that provides a numba model run (with warm-up)"""

    def model_run_numba():
        params = lm.abm.ABMParams(num_ticks=100, verbose=False, use_numba=True, show_progress=False)
        model = lm.abm.ABMModel(test_scenario_df, params)
        model.components = [lm.abm.components.ImportationPressureProcess, lm.abm.components.InfectionProcess]
        model.run()
        return model

    # Warm up numba (JIT compilation)
    model_run_numba()
    return model_run_numba


@pytest.fixture
def python_model(test_scenario_df):
    """Fixture that provides a python model run"""

    def model_run_python():
        params = lm.abm.ABMParams(num_ticks=100, verbose=False, use_numba=False, show_progress=False)
        model = lm.abm.ABMModel(test_scenario_df, params)
        model.components = [lm.abm.components.ImportationPressureProcess, lm.abm.components.InfectionProcess]
        model.run()
        return model

    return model_run_python


def test_numba_model_produces_infections(numba_model):
    """Test that numba model produces expected results"""
    model = numba_model()
    assert model.patches.states.R.sum() > 0


def test_python_model_produces_infections(python_model):
    """Test that python model produces expected results"""
    model = python_model()
    assert model.patches.states.R.sum() > 0


@pytest.mark.slow
def test_numba_performance_improvement(numba_model, python_model):
    """Test that numba version is faster than python version"""
    nb_time = timeit.timeit(numba_model, number=10)
    python_time = timeit.timeit(python_model, number=10)

    print(f"Numba time: {nb_time:.2f} seconds")
    print(f"Python time: {python_time:.2f} seconds")
    print(f"Speedup: {python_time / nb_time:.1f}x")

    assert python_time > nb_time, f"Numba should be faster: {nb_time:.2f}s vs {python_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
