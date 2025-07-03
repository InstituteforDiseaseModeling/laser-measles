import numpy as np
from pydantic import BaseModel
from pydantic import Field

from laser_measles.base import BasePhase, BaseLaserModel
from laser_measles.compartmental.mixing import init_gravity_diffusion
from laser_measles.utils import cast_type


class InfectionParams(BaseModel):
    """Parameters specific to the SEIR infection process component."""
    beta: float = Field(default=1.0, description="Transmission rate per day", gt=0.0)
    sigma: float = Field(default=1.0 / 6.0, description="Progression rate from exposed to infectious (1/incubation_period)", gt=0.0)
    gamma: float = Field(default=1.0 / 8.0, description="Recovery rate from infection (1/infectious_period)", gt=0.0)
    seasonality: float = Field(default=0.0, description="Seasonality factor, default is no seasonality", ge=0.0, le=1.0)
    season_start: float = Field(default=0, description="Season start day (0-364)", ge=0, le=364)
    distance_exponent: float = Field(default=1.5, description="Distance exponent", ge=0.0)
    mixing_scale: float = Field(default=0.001, description="Mixing scale", ge=0.0)

    @property
    def basic_reproduction_number(self) -> float:
        """Calculate R0 = beta / gamma"""
        return self.beta / self.gamma

    @property
    def incubation_period(self) -> float:
        """Average incubation period in days"""
        return 1.0 / self.sigma

    @property
    def infectious_period(self) -> float:
        """Average infectious period in days"""
        return 1.0 / self.gamma


class InfectionProcess(BasePhase):
    """
    Component for simulating SEIR disease progression with daily timesteps.

    This class implements a stochastic SEIR infection process that models disease transmission
    and progression through compartments. It uses daily rates and accounts for mixing between
    different population groups.

    The SEIR infection process follows these steps:
    1. Calculate force of infection based on:
       - Base transmission rate (beta)
       - Seasonal variation
       - Population mixing matrix
       - Current number of infectious individuals
    2. Stochastic transitions using binomial sampling:
       - S → E: New exposures based on force of infection
       - E → I: Progression from exposed to infectious
       - I → R: Recovery from infection
    3. Update population states for all compartments

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
    The infection process uses daily rates and seasonal transmission that varies
    sinusoidally over time with a period of 365 days.
    """

    def __init__(self, model: BaseLaserModel, verbose: bool = False, params: InfectionParams | None = None) -> None:
        super().__init__(model, verbose)
        if params is None:
            params = InfectionParams()
        self.params = params
        self._mixing = None

    def __call__(self, model: BaseLaserModel, tick: int) -> None:
        # Get state counts: states is (4, num_patches) for [S, E, I, R]
        states = model.patches.states

        # Calculate total population per patch
        total_pop = states.sum(axis=0)

        # Avoid division by zero
        total_pop = np.maximum(total_pop, 1)

        # Calculate prevalence of infectious individuals in each patch
        prevalence = states.I / total_pop  # I_j / N_j

        # Calculate force of infection with seasonal variation
        seasonal_factor = 1 + self.params.seasonality * np.sin(2 * np.pi * (tick - self.params.season_start) / 365.0)
        lambda_i = (
            self.params.beta * seasonal_factor * np.matmul(self.mixing, prevalence)  # M @ (I_j / N_j)
        )

        # Stochastic transitions using binomial sampling

        # 1. S → E: New exposures
        prob_exposure = 1 - np.exp(-lambda_i)
        new_exposures = np.random.binomial(states.S, prob_exposure).astype(states.dtype)

        # 2. E → I: Progression to infectious
        prob_infection = 1 - np.exp(-self.params.sigma)
        new_infections = np.random.binomial(states.E, prob_infection).astype(states.dtype)

        # 3. I → R: Recovery
        prob_recovery = 1 - np.exp(-self.params.gamma)
        new_recoveries = np.random.binomial(states.I, prob_recovery).astype(states.dtype)

        # Update compartments
        states.S -= new_exposures  # S decreases
        states.E += new_exposures  # E increases
        states.E -= new_infections  # E decreases
        states.I += new_infections  # I increases
        states.I -= new_recoveries  # I decreases
        states.R += new_recoveries  # R increases

        return

    def initialize(self, model: BaseLaserModel) -> None:
        """Initializes the mixing component"""
        self.mixing = init_gravity_diffusion(model.scenario, self.params.mixing_scale, self.params.distance_exponent)

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
