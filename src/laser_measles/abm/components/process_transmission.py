"""
This module defines the Transmission class, which models the transmission of measles in a population.

Classes:

    Transmission: A class to model the transmission dynamics of measles within a population.

Functions:

    Transmission.__init__(self, model, verbose: bool = False) -> None:

        Initializes the Transmission object with the given model and verbosity.

    Transmission.__call__(self, model, tick) -> None:

        Executes the transmission dynamics for a given model and tick.

    nb_transmission_update(susceptibilities, nodeids, forces, etimers, count, exp_shape, exp_scale, incidence):

        A Numba-compiled function to update the transmission dynamics in parallel.

    Transmission.plot(self, fig: Figure = None):

        Plots the cases and incidence for the two largest patches in the model.
"""

import numba as nb
import numpy as np
from pydantic import BaseModel
from pydantic import Field

from laser_measles.abm.model import ABMModel
from laser_measles.base import BasePhase
from laser_measles.compartmental.mixing import init_gravity_diffusion  # TODO: consolidate spatial mixing into separate module
from laser_measles.utils import cast_type


@nb.njit(
    (nb.uint8[:], nb.uint16[:], nb.uint8[:], nb.float64[:], nb.uint16[:], nb.uint32, nb.float32, nb.float32, nb.uint32[:]),
    parallel=True,
    nogil=True,
)
def nb_lognormal_update(states, patch_ids, state, forces, etimers, count, exp_mu, exp_sigma, flow):  # pragma: no cover
    """Numba compiled function to stochastically transmit infection to agents in parallel."""
    max_node_id = np.max(patch_ids)
    thread_incidences = np.zeros((nb.get_num_threads(), max_node_id + 1), dtype=np.uint32)

    for i in nb.prange(count):
        state = states[i]
        if state == 0:
            patch_id = patch_ids[i]
            force = forces[patch_id]  # force of infection attenuated by personal susceptibility
            if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                states[i] = 1  # set state to exposed
                # set exposure timer for newly infected individuals to a draw from a lognormal distribution, must be at least 1 day
                etimers[i] = np.uint16(np.maximum(1, np.round(np.random.lognormal(exp_mu, exp_sigma))))
                thread_incidences[nb.get_thread_id(), patch_id] += 1

    flow[:] = thread_incidences.sum(axis=0)

    return


class TransmissionParams(BaseModel):
    """Parameters specific to the transmission process component."""

    beta: float = Field(default=1.0, description="Base transmission rate", gt=0.0)
    seasonality_factor: float = Field(default=1.0, description="Seasonality factor", ge=0.0, le=1.0)
    season_start: float = Field(default=0.0, description="Seasonality phase", ge=0, le=364)
    exp_mu: float = Field(default=6.0, description="Exposure mean (days)", gt=0.0)
    exp_sigma: float = Field(default=2.0, description="Exposure sigma (days)", gt=0.0)
    distance_exponent: float = Field(default=1.5, description="Distance exponent", ge=0.0)
    mixing_scale: float = Field(default=0.001, description="Mixing scale", ge=0.0)

    @property
    def mu_underlying(self) -> float:
        """The mean of the underlying lognormal distribution."""
        return np.log(self.exp_mu**2 / np.sqrt(self.exp_mu**2 + self.exp_sigma**2))

    @property
    def sigma_underlying(self) -> float:
        """The standard deviation of the underlying lognormal distribution."""
        return np.sqrt(np.log(1 + (self.exp_sigma / self.exp_mu) ** 2))


class TransmissionProcess(BasePhase):
    """
    A component to model the transmission of disease in a population.
    """

    def __init__(self, model, verbose: bool = False, params: TransmissionParams | None = None) -> None:
        """
        Initializes the transmission object.

        Args:

            model: The model object that contains the patches and parameters.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model object passed during initialization.

        The model's patches are extended with the following properties:

            - 'cases': A vector property with length equal to the number of ticks, dtype is uint32.
            - 'forces': A scalar property with dtype float32.
            - 'incidence': A vector property with length equal to the number of ticks, dtype is uint32.
        """

        super().__init__(model, verbose)

        self.params = params if params is not None else TransmissionParams()
        self._mixing = None

        # add new properties to the laserframes
        model.people.add_scalar_property("etimer", dtype=np.uint16, default=0)  # exposure timer
        model.people.add_scalar_property("itimer", dtype=np.uint16, default=0)  # infection timer
        model.patches.add_scalar_property("incidence", dtype=np.uint32, default=0)  # new infections per time step
        return

    def __call__(self, model, tick) -> None:
        """
        Simulate the transmission of measles for a given model at a specific tick.

        This method updates the state of the model by simulating the spread of disease
        through the population and patches. It calculates the contagion, handles the
        migration of infections between patches, and updates the forces of infection
        based on the effective transmission rate and seasonality factors. Finally, it
        updates the infected state of the population.

        Parameters:

            model (object): The model object containing the population, patches, and parameters.
            tick (int): The current time step in the simulation.

        Returns:

            None

        """
        # access the patch and people laserframes
        patches = model.patches
        people = model.people

        seasonal_factor = 1 + self.params.seasonality_factor * np.sin(2 * np.pi * (tick - self.params.season_start) / 365)
        beta_effective = self.params.beta * seasonal_factor

        # transfer between and w/in patches
        # NB: this assumes that the mixing matrix is properly normalized
        # i.e., that the sum of each row is 1 (self.mixing.sum(axis=1) == 1)
        forces = self.mixing @ (beta_effective * patches.states.I)

        # normalize by the population (excluding D state)
        forces /= patches.states[:-1, :].sum(axis=0)
        np.negative(forces, out=forces)
        np.expm1(forces, out=forces) # exp(x) - 1
        np.negative(forces, out=forces)

        # S --> E
        nb_lognormal_update(
            people.state,
            people.patch_id,
            people.state,
            forces,
            people.etimer,
            people.count,
            np.float32(self.params.mu_underlying),
            np.float32(self.params.sigma_underlying),
            model.patches.incidence, # flow
        )
        # Update susceptible and exposed counters
        patches.states.S -= model.patches.incidence
        patches.states.E += model.patches.incidence
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

    def infect(self, model: ABMModel, idx: np.ndarray | int) -> None:
        if isinstance(idx, int):
            idx = np.array([idx])
        people = model.people
        people.state[idx] = 1
        people.etimer[idx] = cast_type(np.maximum(1, np.round(np.random.lognormal(self.params.mu_underlying, self.params.sigma_underlying, size=len(idx)))), people.etimer.dtype)
        return
