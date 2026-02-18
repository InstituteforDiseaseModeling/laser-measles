from abc import ABC
from abc import abstractmethod
from typing import TypeVar

import numpy as np
from pydantic import BaseModel
from pydantic import Field

from laser_measles.base import BasePhase
from laser_measles.utils import cast_type

ModelType = TypeVar("ModelType")


class BaseVitalDynamicsParams(BaseModel):
    """Parameters specific to vital dynamics."""

    crude_birth_rate: float = Field(default=20.0, description="Annual crude birth rate per 1000 population", ge=0.0)
    crude_death_rate: float = Field(default=8.0, description="Annual crude death rate per 1000 population", ge=0.0)
    mcv1_efficacy: float = Field(default=0.9, description="Efficacy of MCV1", ge=0.0, le=1.0)


class BaseVitalDynamicsProcess(BasePhase, ABC):
    """
    Phase for simulating the vital dynamics in the model with MCV1.

    This phase handles the simulation of births and deaths in the population model along
    with routine vaccination (MCV1).

    Parameters
    ----------
    model : object
        The simulation model containing nodes, states, and parameters
    verbose : bool, default=False
        Whether to print verbose output during simulation
    params : VitalDynamicsParams | None, default=None
        Component-specific parameters. If None, will use default parameters

    Notes
    -----
    - Birth rates are calculated per tick
    """

    def __init__(self, model, verbose: bool = False, params: BaseVitalDynamicsParams | None = None) -> None:
        super().__init__(model, verbose)
        if params is None:
            params = BaseVitalDynamicsParams()
        self.params = params

    @property
    def lambda_birth(self) -> float:
        """birth rate per tick"""
        return (1 + self.params.crude_birth_rate / 1000) ** (1 / 365 * self.model.params.time_step_days) - 1

    @property
    def mu_death(self) -> float:
        """death rate per tick"""
        return (1 + self.params.crude_death_rate / 1000) ** (1 / 365 * self.model.params.time_step_days) - 1

    def __call__(self, model, tick: int) -> None:
        # state counts
        states = model.patches.states  # num_compartments x num_patches

        # Vital dynamics
        population = states.sum(axis=0)
        avg_births = population * self.lambda_birth
        vaccinated_births = cast_type(
            model.prng.poisson(avg_births * np.array(model.scenario["mcv1"]) * self.params.mcv1_efficacy), states.dtype
        )  # vaccinated AND protected
        unvaccinated_births = cast_type(
            model.prng.poisson(avg_births * (1 - np.array(model.scenario["mcv1"]) * self.params.mcv1_efficacy)), states.dtype
        )

        avg_deaths = states * self.mu_death
        deaths = cast_type(model.prng.poisson(avg_deaths), states.dtype)  # number of deaths

        states.S += unvaccinated_births  # add births to S
        states.R += vaccinated_births  # add births to R
        states -= deaths  # remove deaths from each compartment

        # make sure that all states >= 0
        np.maximum(states, 0, out=states)

    @abstractmethod
    def calculate_capacity(self, model) -> int:
        """
        Calculate the capacity of the model.
        """
        raise NotImplementedError("No capacity for this model")
