import importlib

import numpy as np
import polars as pl
import pytest
from laser_core import PropertySet

import laser_measles as lm
from laser_measles.base import BaseLaserModel
from laser_measles.base import BasePhase

MEASLES_MODULES = ["laser_measles.biweekly", "laser_measles.compartmental"]
MEASLES_MODULES = ["laser_measles.compartmental"]
SEED = np.random.randint(1000000)


def SI_logistic(t: int, size: int, beta: float, t0: int = 0, i0: int = 1) -> float:
    """
    SI model with logistic growth.

    Args:
        t (int): The time step (days).
        beta (float): The transmission rate (infections per day).
        size (int): The population size (people).
        t0 (int): The time step at which the logistic growth starts.
        i0 (int): The initial number of infected individuals.

    Returns:
        float: The number of infected individuals at time t.
    """
    return size / (1 + (size / i0 - 1) * np.exp(-beta * (t - t0)))


# def half_life(f, **kwargs):
# return sp.optimize.minimize(lambda t: np.abs(f(t, **kwargs)/kwargs["size"] - 0.5), x0=[10])
def SI_logistic_half_life(size: int, beta: float, i0: int = 1) -> float:
    return 1 / beta * np.log(size / i0 - 1)


class LogisticGrowthTrackerBase(BasePhase):
    """
    Tracks the logistic growth of the infected population.
    """

    def __init__(self, model: BaseLaserModel, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        self.tracker = np.zeros(model.params.num_ticks)

    def __call__(self, model: BaseLaserModel, tick: int) -> None:
        for instance in model.instances:
            if hasattr(instance, "params"):
                if hasattr(instance.params, "beta"):
                    beta = instance.params.beta
                    t = model.time_elapsed(units="days")
                    self.tracker[tick] = SI_logistic(t=t, size=self.total_population(), beta=beta)
                    return
        raise ValueError("No beta found in model instances")

    def initialize(self, model: BaseLaserModel) -> None:
        pass

    def total_population(self) -> int:
        """Returns the population size."""
        raise NotImplementedError("Subclasses must implement this method")


class LogisticGrowthTracker(LogisticGrowthTrackerBase):
    """
    Tracks the logistic growth of the infected population.
    """

    def total_population(self) -> int:
        return self.model.patches.states.sum()


class ConvertToSI(BasePhase):
    """
    Converts SEIR model to SI by removing E and R compartments.
    Works for both biweekly and compartmental models.
    """

    def __call__(self, model: BaseLaserModel, tick: int) -> None:
        states = model.patches.states
        if states.shape[0] == 3:  # Biweekly: S, I, R
            states[1] += states[2]  # Move R to I
            states[2] = 0
        elif states.shape[0] == 4:  # Compartmental: S, E, I, R
            states[2] += states[1]  # Move E to I
            states[2] += states[3]  # Move R to I
            states[1] = 0
            states[3] = 0

    def initialize(self, model: BaseLaserModel) -> None:
        pass


@pytest.mark.slow
@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_no_vital_dynamics(measles_module):
    """
    Test logistic growth for SI model with no vital dynamics.
    https://github.com/InstituteforDiseaseModeling/laser-generic/blob/main/notebooks/01_SI_nobirths_logistic_growth.ipynb
    """
    MeaslesModel = importlib.import_module(measles_module)

    problem_params = PropertySet(
        {
            "population_size": 100_000_000,
            "beta": 2 / 14,
            "num_days": 730,  # in days
            "initial_infections": 1,
        }
    )
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
        num_ticks = int(np.ceil(problem_params["num_days"] / 365 * 26))
    else:
        num_ticks = problem_params["num_days"]

    params = MeaslesModel.Params(num_ticks=num_ticks, start_time="2001-01", seed=SEED)

    # Create model
    model = MeaslesModel.Model(params=params, scenario=scenario)

    transmission_params = MeaslesModel.components.InfectionParams(beta=problem_params["beta"])
    seeding_params = MeaslesModel.components.InfectionSeedingParams(num_infections=problem_params["initial_infections"])
    model.components = [
        MeaslesModel.components.StateTracker,
        lm.create_component(MeaslesModel.components.InfectionSeedingProcess, params=seeding_params),
        lm.create_component(MeaslesModel.components.InfectionProcess, params=transmission_params),
        ConvertToSI,
    ]

    # run model
    model.run()

    # Find StateTracker instance
    state_tracker = model.get_instance(MeaslesModel.components.StateTracker)[0]

    # Time to half the population is infectious
    t_2_theory = SI_logistic_half_life(
        size=problem_params["population_size"], beta=problem_params["beta"], i0=problem_params["initial_infections"]
    )
    t_2_simulated = np.interp(
        0.5 * problem_params["population_size"], state_tracker.I, model.params.time_step_days * np.arange(model.params.num_ticks)
    )

    rel_error = (t_2_simulated - t_2_theory) / t_2_theory

    # Different error tolerances for different model types
    if "compartmental" in measles_module:
        assert rel_error < 0.15, f"Relative error: {rel_error} (max 0.15)"
    elif "biweekly" in measles_module:
        assert rel_error < 0.25, f"Relative error: {rel_error} (max 0.25)"

    print(f"t_2_theory: {t_2_theory}, t_2_sim: {t_2_simulated}")
    return (t_2_simulated - t_2_theory) / t_2_theory


if __name__ == "__main__":
    for module in MEASLES_MODULES:
        print(f"Testing {module}...")
        rel_error = test_no_vital_dynamics(module)
        print(f"✓ {module} test passed with relative error: {rel_error:.4f}")
