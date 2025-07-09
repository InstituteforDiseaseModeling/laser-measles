"""
This module defines the MaternalAntibodies class and the nb_update_ma_timers function for simulating the presence of maternal antibodies in a population model.

Classes:

    MaternalAntibodies: Manages the maternal antibodies for a population model, including initialization, updates, and plotting.

Usage:

    The MaternalAntibodies class should be instantiated with a model object and can be called to update the model at each tick. It also provides a method to handle newborns and a method to plot the current state of maternal antibodies in the population.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class MaternalAntibodies:
    """
    A component to manage maternal antibodies in a population model.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize the maternal antibodies component of the model.

        Args:

            model: The model instance to which this component belongs.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model instance to which this component belongs.

        Notes:

            This initializer also adds a scalar property "ma_timer" to the model's population,
            which is used to track the maternal antibodies timer for each agent.
        """

        self.model = model

        # TODO - initialize existing agents with maternal antibodies
        model.population.add_scalar_property("ma_timer", np.uint8)  # Use uint8 for timer since 6 months ~ 180 days < 2^8

        return

    def __call__(self, model, tick) -> None:
        """
        Updates maternal antibody timers and susceptibility for the population.

        Args:

            model: The model containing the population data.
            tick: The current time step or tick in the simulation.

        Returns:

            None
        """

        self.nb_update_ma_timers(model.population.count, model.population.ma_timer, model.population.susceptibility)
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:]), parallel=True, cache=True)
    def nb_update_ma_timers(count, ma_timers, susceptibility):  # pragma: no cover
        """Numba compiled function to check and update maternal antibody timers for the population in parallel."""
        for i in nb.prange(count):
            timer = ma_timers[i]
            if timer > 0:
                timer -= 1
                ma_timers[i] = timer
                if timer == 0:
                    susceptibility[i] = 1

        return

    def on_birth(self, model, _tick, istart, iend) -> None:
        """
        This function is called when births occur in the model. It updates the susceptibility and maternal antibody timers for newborns.

        Parameters:

            model (object): The model instance containing the population data.
            tick (int): The current tick or time step in the simulation (unused in this function).
            istart (int): The starting index of the newborns in the population array.
            iend (int): The ending index of the newborns in the population array.

        Returns:

            None
        """

        model.population.susceptibility[istart:iend] = 0  # newborns are _not_ susceptible due to maternal antibodies
        model.population.ma_timer[istart:iend] = int(6 * 365 / 12)  # 6 months in days
        return

    def plot(self, fig: Figure = None):
        """
        Plots a pie chart showing the distribution of infants with and without maternal antibodies.

        Parameters:

            fig (Figure, optional): A Matplotlib Figure object. If None, a new figure will be created with default size and DPI.

        Returns:

            None
        """

        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        cinfants = ((self.model.params.nticks - self.model.population.dob[0 : self.model.population.count]) < 365).sum()
        cwith = (self.model.population.ma_timer[0 : self.model.population.count] > 0).sum()
        cwithout = cinfants - cwith

        fig.suptitle(f"Maternal Antibodies for Infants (< 1 year)\n{cinfants:,} Infants")
        plt.pie([cwithout, cwith], labels=[f"Infants w/out Antibodies {cwithout:,}", f"Infants w/Maternal Antibodies {cwith:,}"])

        yield
        return
