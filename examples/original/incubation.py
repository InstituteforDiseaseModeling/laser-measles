"""
This module defines the Incubation class, which simulates the incubation period of a disease within a population model.

Classes:
    Incubation: Manages the incubation period of a disease, updating exposure timers and handling birth events.

Methods:
    __init__(self, model, verbose: bool = False) -> None:
        Initializes the Incubation instance with the given model and optional verbosity.

    __call__(self, model, tick) -> None:
        Updates the exposure timers for the population at each tick of the simulation.

    nb_update_exposure_timers(count, etimers, itimers, inf_mean, inf_std) -> None:
        Numba-optimized static method to update exposure timers and set infection timers when exposure ends.

    on_birth(self, model, _tick, istart, iend) -> None:
        Handles the birth event by setting the exposure timers of newborns to zero.

    nb_set_etimers(istart, iend, incubation, value) -> None:
        Numba-optimized static method to set exposure timers for a range of individuals.

    plot(self, fig: Figure = None):
        Plots the distribution of the incubation period using matplotlib.

Attributes:
    model: The population model associated with the incubation period.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Incubation:
    """
    A component to update the incubation timers of a population in a model.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize the incubation process for the model.

        Args:

            model: The model instance to which the incubation process is being added.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model instance passed to the constructor.

        Notes:

            Adds a scalar property "etimer" to the model's population with a default value of 0.
            The "etimer" property is of type uint8.
            There is a TODO to verify the "itimer" property on the population.
        """

        self.model = model

        model.population.add_scalar_property("etimer", dtype=np.uint8, default=0)
        # TODO - verify itimer property on population since we use it when etimer hits 0
        # model.population.add_scalar_property("itimer", dtype=np.uint8, default=0)

        return

    def __call__(self, model, tick) -> None:
        """
        Updates the exposure timers for the population in the model at each tick.

        Args:

            model: The simulation model containing the population and parameters.
            tick: The current time step in the simulation.

        Returns:

            None
        """

        Incubation.nb_update_exposure_timers(
            model.population.count, model.population.etimer, model.population.itimer, model.params.inf_mean, model.params.inf_std
        )
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:], nb.float32, nb.float32), parallel=True, cache=True)
    def nb_update_exposure_timers(count, etimers, itimers, inf_mean, inf_std) -> None:  # pragma: no cover
        """Numba compiled function to check and update exposure timers for the population in parallel."""
        for i in nb.prange(count):
            timer = etimers[i]
            if timer > 0:
                timer -= 1
                etimers[i] = timer
                if timer == 0:
                    itimers[i] = np.maximum(np.uint8(1), np.uint8(np.round(np.random.normal(inf_mean, inf_std))))

        return

    def on_birth(self, model, _tick, istart, iend) -> None:
        """
        This function is called when a birth event occurs in the model. It sets the incubation timer
        for the newborns to zero, indicating that they are not incubating/exposed.

        Parameters:

            model (object): The model instance containing the population data.
            tick (int): The current tick or time step in the simulation (unused in this function).
            istart (int): The start index of the newborns in the population array.
            iend (int): The end index of the newborns in the population array.

        Returns:

            None
        """

        # newborns are _not_ incubating
        # Incubation.nb_set_etimers(istart, iend, model.population.etimer, 0)
        model.population.etimer[istart:iend] = 0

        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint8[:], nb.uint8), parallel=True, cache=True)
    def nb_set_etimers(istart, iend, incubation, value) -> None:  # pragma: no cover
        """Numba compiled function to set exposure timers for a range of agents in parallel."""
        for i in nb.prange(istart, iend):
            incubation[i] = value

        return

    def plot(self, fig: Figure = None):
        """
        Plots the incubation timer values for currently incubating (exposed) agents in the population.

        Parameters:

            fig (Figure, optional): A Matplotlib Figure object. If None, a new figure will be created with default size and DPI.

        Yields:

            None: This function uses a generator to yield control back to the caller.
        """

        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Incubation Period Distribution")

        etimers = self.model.population.etimer[0 : self.model.population.count]
        incubating = etimers > 0
        incubation_counts = np.bincount(etimers[incubating])
        plt.bar(range(len(incubation_counts)), incubation_counts)

        yield
        return
