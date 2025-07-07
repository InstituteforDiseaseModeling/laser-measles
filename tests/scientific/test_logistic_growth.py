import numpy as np
import polars as pl
import pytest
from laser_core import PropertySet

import laser_measles as lm
from laser_measles.base import BaseLaserModel
from laser_measles.base import BasePhase
from laser_measles.biweekly import BiweeklyModel
from laser_measles.compartmental import CompartmentalModel


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


class LogisticGrowthBiweeklyTracker(LogisticGrowthTrackerBase):
    """
    Tracks the logistic growth of the infected population in the biweekly model.
    """

    def total_population(self) -> int:
        return self.model.patches.states.sum()


class Biweekly2SI(BasePhase):
    """
    Removes recovereds from the model
    """

    def __call__(self, model: BiweeklyModel, tick: int) -> None:
        model.patches.states[1] += model.patches.states[2]
        model.patches.states[2] = 0

    def initialize(self, model: BiweeklyModel) -> None:
        pass


class Compartmental2SI(BasePhase):
    """
    Removes recovereds from the model
    """

    def __call__(self, model: CompartmentalModel, tick: int) -> None:
        model.patches.states[2] += model.patches.states[1]
        model.patches.states[2] += model.patches.states[3]
        model.patches.states[3] = 0
        model.patches.states[1] = 0

    def initialize(self, model: CompartmentalModel) -> None:
        pass


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["compartmental", "biweekly"])
def test_no_vital_dynamics(model_type):
    """
    Test logistic growth for SI model with no vital dynamics.
    https://github.com/InstituteforDiseaseModeling/laser-generic/blob/main/notebooks/01_SI_nobirths_logistic_growth.ipynb
    """
    problem_params = PropertySet(
        {
            "population_size": 100_000_000,
            "beta": 2 / 14,
            "num_ticks": 730,  # in days
            "initial_infections": 1,
        }
    )
    scenario = pl.DataFrame(
        {
            "id": ["node_0"],
            "pop": [problem_params["population_size"]],
        }
    )

    if model_type == "biweekly":
        # create params
        params = lm.biweekly.BiweeklyParams(
            num_ticks=(problem_params["num_ticks"] // 365) * 26, start_time="2001-01", seed=np.random.randint(1000000)
        )
        # create model
        model = lm.biweekly.Model(params=params, scenario=scenario)
        # seed with infections
        model.infect(indices=0, num_infected=problem_params["initial_infections"])
        # Add components
        transmission_params = lm.biweekly.components.InfectionParams(beta=problem_params["beta"])
        model.components = [
            lm.biweekly.components.StateTracker,
            lm.create_component(lm.biweekly.components.InfectionProcess, params=transmission_params),
            Biweekly2SI,
            LogisticGrowthBiweeklyTracker,
        ]
    elif model_type == "compartmental":
        # create params
        params = lm.compartmental.CompartmentalParams(
            num_ticks=problem_params["num_ticks"], start_time="2001-01", seed=np.random.randint(1000000)
        )
        # create model
        model = lm.compartmental.Model(params=params, scenario=scenario)
        # seed with infections
        model.expose(indices=0, num_exposed=problem_params["initial_infections"])
        model.infect(indices=0, num_infected=problem_params["initial_infections"])
        # add transmission params
        transmission_params = lm.compartmental.components.InfectionParams(beta=problem_params["beta"])
        model.components = [
            lm.compartmental.components.StateTracker,
            lm.create_component(lm.compartmental.components.InfectionProcess, params=transmission_params),
            Compartmental2SI,
        ]
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # run model
    model.run()

    for instance in model.instances:
        if isinstance(instance, lm.biweekly.components.StateTracker | lm.compartmental.components.StateTracker):
            state_tracker = instance

    # Time to half the population is infectious
    t_2_theory = SI_logistic_half_life(
        size=problem_params["population_size"], beta=problem_params["beta"], i0=problem_params["initial_infections"]
    )
    t_2_simulated = np.interp(
        0.5 * problem_params["population_size"], state_tracker.I, model.params.time_step_days * np.arange(model.params.num_ticks)
    )

    rel_error = (t_2_simulated - t_2_theory) / t_2_theory

    # Different error tolerances for different model types
    if model_type == "compartmental":
        assert rel_error < 0.15, f"Relative error: {rel_error} (max 0.15)"
    elif model_type == "biweekly":
        assert rel_error < 0.25, f"Relative error: {rel_error} (max 0.25)"

    print(f"t_2_theory: {t_2_theory}, t_2_sim: {t_2_simulated}")
    return (t_2_simulated - t_2_theory) / t_2_theory


if __name__ == "__main__":
    rel_errors = []
    for i in range(1):
        rel_error = test_no_vital_dynamics("compartmental")
        rel_errors.append(rel_error)
    print(f"Relative errors: {rel_errors}")
    print(f"Mean relative error: {np.mean(rel_errors)}")
    print(f"Std relative error: {np.std(rel_errors)}")
