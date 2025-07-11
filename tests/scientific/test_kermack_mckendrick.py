"""
Kermack-McKendrick final outbreak size test:

Tests the final outbreak size against the theoretical Kermack-McKendrick prediction
for SIR/SEIR models. The theoretical final outbreak size z satisfies:
z = 1 - exp(-R0 * z)

This test validates that the model simulations converge to the theoretical
final outbreak size for epidemic scenarios.
"""
# ruff: noqa: PT006

import importlib

import numpy as np
import polars as pl
import pytest
from laser_core import PropertySet
from scipy.optimize import fsolve

import laser_measles as lm

MEASLES_MODULES = ["laser_measles.abm", "laser_measles.biweekly", "laser_measles.compartmental"]
SEED = np.random.randint(1000000)
RNG = np.random.default_rng(SEED)


def SIR_final_outbreak_size(R0: float) -> float:
    """
    Calculate the theoretical final outbreak size for an SIR epidemic.

    Solves the implicit equation: z = 1 - exp(-R0 * z)
    where z is the fraction of the population that gets infected.

    Args:
        R0: Basic reproduction number

    Returns:
        Final outbreak size as fraction of population
    """
    if R0 <= 1.0:
        return 0.0

    # Initial guess for z
    z0 = 1.0 - 1.0 / R0  # Rough approximation

    # Solve z = 1 - exp(-R0 * z)
    # Rearranged as: z - 1 + exp(-R0 * z) = 0
    def equation(z):
        return z - 1 + np.exp(-R0 * z)

    solution = fsolve(equation, z0)[0]
    return max(0.0, min(1.0, solution))  # Clamp to [0, 1]


def single_test(MeaslesModel, problem_params, measles_module):
    """
    Run a single test for final outbreak size validation.
    """
    scenario = pl.DataFrame(
        {
            "id": ["node_0"],
            "pop": [problem_params["population_size"]],
            "lat": [40.0],
            "lon": [4.0],
            "mcv1": [0.0],
        }
    )

    # Create model-specific parameters
    if "biweekly" in measles_module:
        num_ticks = int(np.ceil(problem_params["num_days"] / 14))
    else:
        num_ticks = problem_params["num_days"]

    params = MeaslesModel.Params(num_ticks=num_ticks, start_time="2001-01", seed=RNG.integers(1000000))

    # Create model
    model = MeaslesModel.Model(params=params, scenario=scenario)

    # Set up components for epidemic simulation
    transmission_params = MeaslesModel.components.InfectionParams(beta=problem_params["beta"])
    seeding_params = MeaslesModel.components.InfectionSeedingParams(num_infections=problem_params["initial_infections"])

    model.components = [
        MeaslesModel.components.StateTracker,
        lm.create_component(MeaslesModel.components.InfectionSeedingProcess, params=seeding_params),
        lm.create_component(MeaslesModel.components.InfectionProcess, params=transmission_params),
    ]

    # Run model
    model.run()

    # Find StateTracker instance
    state_tracker = model.get_instance("StateTracker")[0]

    # Calculate final outbreak size
    initial_S = state_tracker.S[0]
    final_S = state_tracker.S[-1]
    simulated_final_size = (initial_S - final_S) / initial_S

    # Calculate theoretical final outbreak size
    theoretical_final_size = SIR_final_outbreak_size(problem_params["R0"])

    # Calculate relative error
    if theoretical_final_size > 0:
        rel_error = abs(simulated_final_size - theoretical_final_size) / theoretical_final_size
    else:
        rel_error = abs(simulated_final_size)

    return rel_error


@pytest.mark.slow
@pytest.mark.parametrize(
    "measles_module,num_reps", [("laser_measles.abm", 5), ("laser_measles.biweekly", 5), ("laser_measles.compartmental", 5)]
)
def test_final_outbreak_size(measles_module, num_reps):
    """
    Test final outbreak size against Kermack-McKendrick theoretical prediction.

    This test simulates a measles-like epidemic and compares the final outbreak
    size with the theoretical prediction from the Kermack-McKendrick model.
    """
    MeaslesModel = importlib.import_module(measles_module)

    # Measles-like epidemic parameters
    infectious_period = 14  # days
    R0 = 7.0  # typical for measles

    # Calculate transmission rate
    # For SIR: beta = R0 / infectious_period
    # For SEIR: beta = R0 / infectious_period (same, gamma handles the rest)
    beta = R0 / infectious_period

    problem_params = PropertySet(
        {
            "population_size": 100_000,
            "beta": beta,
            "R0": R0,
            "num_days": 730,  # 2 years, enough for epidemic to complete
            "initial_infections": 10,
        }
    )

    rel_errors = []
    for _ in range(num_reps):
        rel_errors.append(single_test(MeaslesModel, problem_params, measles_module))

    mean_error = np.mean(rel_errors)
    std_error = np.std(rel_errors)

    print(f"Relative error: {mean_error:.4f} ± {std_error:.4f}")

    # Different error tolerances for different model types
    if "abm" in measles_module:
        assert mean_error < 0.03, f"ABM relative error: {mean_error:.4f} (max 0.03)"
    elif "compartmental" in measles_module:
        assert mean_error < 0.02, f"Compartmental relative error: {mean_error:.4f} (max 0.02)"
    elif "biweekly" in measles_module:
        assert mean_error < 0.05, f"Biweekly relative error: {mean_error:.4f} (max 0.05)"


if __name__ == "__main__":
    for module in MEASLES_MODULES:
        print(f"Testing {module}...")
        test_final_outbreak_size(module, num_reps=10)
        print(f"✓ {module} test passed")
