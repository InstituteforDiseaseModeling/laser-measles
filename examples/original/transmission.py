"""
This module defines the Transmission class, which models the transmission of measles in a population.

Classes:

    Transmission: A class to model the transmission dynamics of measles within a population.

Functions:

    Transmission.__init__(self, model, verbose: bool = False) -> None:

        Initializes the Transmission object with the given model and verbosity.

    Transmission.__call__(self, model, tick) -> None:

        Executes the transmission dynamics for a given model and tick.

    Transmission.nb_transmission_update(susceptibilities, nodeids, forces, etimers, count, exp_shape, exp_scale, incidence):

        A Numba-compiled static method to update the transmission dynamics in parallel.

    Transmission.plot(self, fig: Figure = None):

        Plots the cases and incidence for the two largest patches in the model.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Transmission:
    """
    A component to model the transmission of disease in a population.
    """

    def __init__(self, model, verbose: bool = False) -> None:
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

        self.model = model

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

        contagion = patches.cases[tick, :]  # we will accumulate current infections into this view into the cases array
        nodeids = population.nodeid[0 : population.count]  # just look at the active agent indices
        itimers = population.itimer[0 : population.count]  # just look at the active agent indices
        np.add.at(contagion, nodeids[itimers > 0], 1)  # increment by the number of active agents with non-zero itimer

        network = patches.network
        transfer = (contagion * network).round().astype(np.uint32)
        contagion += transfer.sum(axis=1)  # increment by incoming "migration"
        contagion -= transfer.sum(axis=0)  # decrement by outgoing "migration"

        forces = patches.forces
        beta_effective = model.params.beta + model.params.seasonality_factor * np.sin(
            2 * np.pi * (tick - model.params.seasonality_phase) / 365
        )
        np.multiply(contagion, beta_effective, out=forces)
        np.divide(forces, model.patches.populations[tick, :], out=forces)  # per agent force of infection
        # rate to probability conversion 1 - e^(-rate)
        np.negative(forces, out=forces)
        np.expm1(forces, out=forces)
        np.negative(forces, out=forces)

        Transmission.nb_transmission_update(
            population.susceptibility,
            population.nodeid,
            forces,
            population.etimer,
            population.count,
            model.params.exp_shape,
            model.params.exp_scale,
            model.patches.incidence[tick, :],
        )

        return

    @staticmethod
    @nb.njit(
        (nb.uint8[:], nb.uint16[:], nb.float32[:], nb.uint8[:], nb.uint32, nb.float32, nb.float32, nb.uint32[:]),
        parallel=True,
        nogil=True,
        cache=True,
    )
    def nb_transmission_update(susceptibilities, nodeids, forces, etimers, count, exp_shape, exp_scale, incidence):  # pragma: no cover
        """Numba compiled function to stochastically transmit infection to agents in parallel."""
        for i in nb.prange(count):
            susceptibility = susceptibilities[i]
            if susceptibility > 0:
                nodeid = nodeids[i]
                force = susceptibility * forces[nodeid]  # force of infection attenuated by personal susceptibility
                if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                    susceptibilities[i] = 0.0  # no longer susceptible
                    # set exposure timer for newly infected individuals to a draw from a gamma distribution, must be at least 1 day
                    etimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.gamma(exp_shape, exp_scale))))

                    incidence[nodeid] += 1

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
