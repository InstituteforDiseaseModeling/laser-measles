"""
This module defines the MaternalAntibodies class and the nb_update_ma_timers function for simulating the presence of maternal antibodies in a population model.

Classes:
    MaternalAntibodies: Manages the maternal antibodies for a population model, including initialization, updates, and plotting.

Functions:
    nb_update_ma_timers(count, ma_timers, susceptibility): Numba-optimized function to update maternal antibody timers and susceptibility status for a population.

Usage:
    The MaternalAntibodies class should be instantiated with a model object and can be called to update the model at each tick. It also provides a method to handle newborns and a method to plot the current state of maternal antibodies in the population.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class MaternalAntibodies:
    """
    A class to manage maternal antibodies in a population model.

    Attributes:
    -----------

    model : object

        The population model instance.

    verbose : bool, optional

        If True, enables verbose output (default is False).

    Methods:
    --------

    __call__(model, tick) -> None

        Updates maternal antibody timers and susceptibility for the population at each tick.

    on_birth(model, _tick, istart, iend) -> None

        Sets the susceptibility and maternal antibody timer for newborns.

    plot(fig: Figure = None)

        Plots the distribution of maternal antibodies among infants.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize the maternal antibodies component of the model.
        Args:
            model: The model instance to which this component belongs.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        Attributes:
            __name__ (str): The name of the component, set to "maternal_antibodies".
            model: The model instance to which this component belongs.
        Notes:
            This initializer also adds a scalar property "ma_timer" to the model's population,
            which is used to track the maternal antibodies timer for each agent.
        """

        self.__name__ = "maternal_antibodies"
        self.model = model

        # TODO - initialize existing agents with maternal antibodies
        model.population.add_scalar_property("ma_timer", np.uint8)  # Use uint8 for timer since 6 months ~ 180 days < 2^8

        return

    def __call__(self, model, tick) -> None:
        """
        Updates maternal antibody timers and susceptibility for the population.
        This method is called to update the maternal antibody timers and susceptibility
        of the population within the given model at each tick.

        Args:

            model: The model containing the population data.
            tick: The current time step or tick in the simulation.

        Returns:

            None
        """

        nb_update_ma_timers(model.population.count, model.population.ma_timer, model.population.susceptibility)
        return

    def on_birth(self, model, _tick, istart, iend) -> None:
        """
        Handle the birth event in the model by updating the susceptibility and maternal antibody timer for newborns.

        Parameters:

            model (object): The model instance containing the population data.
            _tick (int): The current tick or time step in the simulation.
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


@nb.njit((nb.uint32, nb.uint8[:], nb.uint8[:]), parallel=True, cache=True)
def nb_update_ma_timers(count, ma_timers, susceptibility):  # pragma: no cover
    for i in nb.prange(count):
        timer = ma_timers[i]
        if timer > 0:
            timer -= 1
            ma_timers[i] = timer
            if timer == 0:
                susceptibility[i] = 1

    return
