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
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, Field
from typing import Optional

from laser_measles.base import BaseComponent

@nb.njit(
    (nb.uint8[:], nb.uint16[:], nb.uint8[:], nb.float32[:], nb.uint16[:], nb.uint32, nb.float32, nb.float32, nb.uint32[:], nb.uint32[:], nb.int_),
    parallel=True,
    nogil=True,
    cache=True,
)
def nb_transmission_update(
    states, nodeids, state, forces, etimers, count, exp_mu, exp_sigma, incidence, doi, tick
):  # pragma: no cover
    """Numba compiled function to stochastically transmit infection to agents in parallel."""
    max_node_id = np.max(nodeids)
    thread_incidences = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1), dtype=np.uint32)

    for i in nb.prange(count):
        state = states[i]
        if state == 0:
            nodeid = nodeids[i]
            force = forces[nodeid]  # force of infection attenuated by personal susceptibility
            if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                states[i] = 1  # set state to exposed
                # set exposure timer for newly infected individuals to a draw from a lognormal distribution, must be at least 1 day
                etimers[i] = np.maximum(np.uint16(1), np.uint16(np.round(np.random.lognormal(exp_mu, exp_sigma))))
                doi[i] = tick
                thread_incidences[nb.get_thread_id(), nodeid] += 1

    incidence[:] = thread_incidences.sum(axis=0)

    return
class TransmissionParams(BaseModel):
    """Parameters specific to the transmission process component."""
    
    beta: float = Field(default=32, description="Base transmission rate", gt=0.0)
    seasonality_factor: float = Field(default=1.0, description="Seasonality factor", ge=0.0, le=1.0)
    seasonality_phase: float = Field(default=0, description="Seasonality phase")
    exp_mu: float = Field(default=11.0, description="Exposure mean (lognormal)", gt=0.0)
    exp_sigma: float = Field(default=2.0, description="Exposure sigma (lognormal)", gt=0.0)
    inf_mean: float = Field(default=8.0, description="Mean infection duration", gt=0.0)
    inf_sigma: float = Field(default=2.0, description="Shape parameter for infection duration", gt=0.0)

    @property
    def inf_shape(self) -> float:
        return (self.inf_mean / self.inf_sigma) ** 2
    
    @property
    def inf_scale(self) -> float:
        return self.inf_mean / self.inf_shape
    
class TransmissionProcess(BaseComponent):
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

        model.population.add_scalar_property("etimer", dtype=np.uint16, default=0)
        model.population.add_scalar_property("itimer", dtype=np.uint16, default=0)
        model.population.add_scalar_property("doi", dtype=np.uint32, default=0)

        model.patches.add_vector_property("exposed", length=model.params.nticks, dtype=np.uint32)
        model.patches.add_vector_property("recovered", length=model.params.nticks, dtype=np.uint32)
        model.patches.add_vector_property("cases", length=model.params.nticks, dtype=np.uint32)
        model.patches.add_scalar_property("forces", dtype=np.float32)
        model.patches.add_vector_property("incidence", model.params.nticks, dtype=np.uint32)

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
        patches = model.patches
        population = model.population

        patches.cases[tick,:] = np.bincount(population.nodeid[population.state == 2], 
                                            minlength=patches.count).astype(patches.populations.dtype)

        contagion = patches.cases[tick, :]  # we will accumulate current infections into this view into the cases array
        # contagion = patches.cases_test[tick, :].copy().astype(np.float32)
        if hasattr(patches, "network"):
            network = patches.network
            transfer = contagion * network.T
            contagion += transfer.sum(axis=1)  # increment by incoming "migration"
            contagion -= transfer.sum(axis=0)  # decrement by outgoing "migration"

        forces = patches.forces
        beta_effective = self.params.beta
        if 'seasonality_factor' in model.params:
            beta_effective *= (1+ self.params.seasonality_factor * np.sin(
            2 * np.pi * (tick - self.params.seasonality_phase) / 365
            ))

        np.multiply(contagion, beta_effective, out=forces)
        np.divide(forces, model.patches.populations, out=forces)  # per agent force of infection
        np.negative(forces, out=forces)
        np.expm1(forces, out=forces)
        np.negative(forces, out=forces)

        nb_transmission_update(
            population.state,
            population.nodeid,
            population.state,
            forces,
            population.etimer,
            population.count,
            self.params.exp_mu,
            self.params.exp_sigma,
            model.patches.incidence[tick, :],
            population.doi,
            tick,
        )
        return


    def on_birth(self, model, _tick, istart, iend) -> None:
        """
        This function sets the date of infection for newborns to zero.
        Appears here because transmission is where I have decided to add the "doi" property,
        and I think it thus makes sense to also have the on-birth initializer here.  Could
        just as easily choose to do this over in Infection class instead.

        Args:

            model: The simulation model containing the population data.
            tick: The current tick or time step in the simulation (unused in this function).
            istart: The starting index of the newborns in the population array.
            iend: The ending index of the newborns in the population array.

        Returns:

            None
        """

        if iend is not None:
            model.population.doi[istart:iend] = 0
        else:
            model.population.doi[istart] = 0
        return

    def plot(self, fig: Figure = None):
        """
        Plots the cases and incidence for the two largest patches in the model.

        This function creates a figure with four subplots:

            - Cases for the largest patch
            - Incidence for the largest patch
            - Cases for the second largest patch
            - Incidence for the second largest patch

        If no figure is provided, a new figure is created with a size of 12x9 inches and a DPI of 128.

        Parameters:

            fig (Figure, optional): A Matplotlib Figure object to plot on. If None, a new figure is created.

        Yields:

            None
        """

        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Cases and Incidence for Two Largest Patches")

        itwo, ione = np.argsort(self.model.patches.populations[-1, :])[-2:]

        fig.add_subplot(2, 2, 1)
        plt.title(f"Cases - Node {ione}")  # ({self.names[ione]})")
        plt.plot(self.model.patches.cases[:, ione])

        fig.add_subplot(2, 2, 2)
        plt.title(f"Incidence - Node {ione}")  # ({self.names[ione]})")
        plt.plot(self.model.patches.incidence[:, ione])

        fig.add_subplot(2, 2, 3)
        plt.title(f"Cases - Node {itwo}")  # ({self.names[itwo]})")
        plt.plot(self.model.patches.cases[:, itwo])

        fig.add_subplot(2, 2, 4)
        plt.title(f"Incidence - Node {itwo}")  # ({self.names[itwo]})")
        plt.plot(self.model.patches.incidence[:, itwo])

        yield
        return