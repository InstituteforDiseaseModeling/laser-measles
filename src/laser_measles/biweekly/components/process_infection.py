import numpy as np
from pydantic import Field

from laser_measles.base import BaseLaserModel
from laser_measles.biweekly.mixing import init_gravity_diffusion
from laser_measles.components import BaseInfection
from laser_measles.components import BaseInfectionParams


class InfectionParams(BaseInfectionParams):
    """Parameters specific to the infection process component."""

    beta: float = Field(
        default=1 * 8 / 14, description="Base transmission rate (infections per day)", ge=0.0
    )  # beta = R0 / (mean infectious period)
    seasonality: float = Field(default=0.0, description="Seasonality factor, default is no seasonality", ge=0.0, le=1.0)
    season_start: int = Field(default=0, description="Season start tick (0-25)", ge=0, le=25)
    distance_exponent: float = Field(default=1.5, description="Distance exponent", ge=0.0)
    mixing_scale: float = Field(default=0.001, description="Mixing scale", ge=0.0)

    @property
    def beta_per_tick(self) -> float:
        return (self.beta * 365) / 26


class InfectionProcess(BaseInfection):
    """
    Component for simulating the spread of infection in the model.

    This class implements a stochastic infection process that models disease transmission
    between different population groups. It uses a seasonally-adjusted transmission rate
    and accounts for mixing between different population groups.

    The infection process follows these steps:
    1. Calculates expected new infections based on:
       - Base transmission rate (beta)
       - Seasonal variation
       - Population mixing matrix
       - Current number of infected individuals
    2. Converts expected infections to probabilities
    3. Samples actual new infections from a binomial distribution
    4. Updates population states:
       - Moves current infected to recovered (configurable recovery period)
       - Adds new infections to infected population
       - Removes new infections from susceptible population

    Parameters
    ----------
    model : object
        The simulation model containing population states and parameters
    verbose : bool, default=False
        Whether to print detailed information during execution
    params : InfectionParams | None, default=None
        Component-specific parameters. If None, will use default parameters

    Notes
    -----
    The infection process uses a configurable recovery period and seasonal
    transmission rate that varies sinusoidally over time.
    """

    def __init__(self, model: BaseLaserModel, verbose: bool = False, params: InfectionParams | None = None) -> None:
        super().__init__(model, verbose)
        if params is None:
            params = InfectionParams()
        self.params = params
        self._mixing = None

    def __call__(self, model: BaseLaserModel, tick: int) -> None:
        # state counts
        states = model.patches.states

        # prevalence in each patch
        prevalence = states.I / states.sum(axis=0)  # I_j / N_j

        lambda_i = (
            self.params.beta_per_tick
            * (1 + self.params.seasonality * np.sin(2 * np.pi * (tick - self.params.season_start) / 26.0))
            * prevalence
        ) @ self.mixing

        prob = 1 - np.exp(-lambda_i)  # already per-susceptible
        dI = model.prng.binomial(states[0], prob).astype(states.dtype)

        # move all currently infected to recovered (using configurable recovery period)
        states[2] += states[1]
        states[1] = 0

        # update susceptible and infected populations
        states[1] += dI  # add new infections to I
        states[0] -= dI  # remove new infections from S

        return

    @property
    def mixing(self) -> np.ndarray:
        """Returns the mixing matrix, initializing if necessary"""
        if self._mixing is None:
            self._mixing = init_gravity_diffusion(self.model.scenario, self.params.mixing_scale, self.params.distance_exponent)
        return self._mixing

    @mixing.setter
    def mixing(self, mixing: np.ndarray) -> None:
        """Sets the mixing matrix"""
        self._mixing = mixing

    def _initialize(self, model: BaseLaserModel) -> None:
        """Initializes the mixing component"""
        self.mixing = init_gravity_diffusion(model.scenario, self.params.mixing_scale, self.params.distance_exponent)
