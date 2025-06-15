"""
This module defines Importation classes, which provide methods to import cases into a population during simulation.

Classes:
    Infect_Random_Agents: A class to periodically infect a random subset of agents in the population

Functions:
    Infect_Random_Agents.__init__(self, model, period, count, start, verbose: bool = False) -> None:
        Initializes the Infect_Random_Agents class with a given model, period, count, and verbosity option.

    Infect_Random_Agents.__call__(self, model, tick) -> None:
        Checks whether it is time to infect a random subset of agents and infects them if necessary.

    Infect_Random_Agents.plot(self, fig: Figure = None):
        Nothing yet.
"""

import numpy as np
from matplotlib.figure import Figure

from ..utils import seed_infections_in_patch, seed_infections_randomly


class InfectRandomAgentsProcess:
    """
    A component to update the infection timers of a population in a model.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize an Infect_Random_Agents instance.

        Args:

            model: The model object that contains the population.
            period: The number of ticks between each infection event.
            count: The number of agents to infect at each event.
            start (int, optional): The tick at which to start the infection events.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model object that contains the population.

        Side Effects:

        """

        self.model = model
        self.period = model.params.importation_period
        self.count = model.params.importation_count
        self.start = 0
        self.end = model.params.nticks
        if hasattr(model.params, "importation_start"):
            self.start = model.params.importation_start
        if hasattr(model.params, "importation_end"):
            self.end = model.params.importation_end

        return

    def __call__(self, model, tick) -> None:
        """
        Updates the infection timers for the population in the model.

        Args:

            model: The model containing the population data.
            tick: The current tick or time step in the simulation.

        Returns:

            None
        """
        if (tick >= self.start) and ((tick - self.start) % self.period == 0) and (tick < self.end):
            inf_nodeids = seed_infections_randomly(model, self.count)
            if hasattr(model.patches, 'cases_test'):
                unique, counts = np.unique(inf_nodeids, return_counts=True)
                for nodeid, count in zip(unique, counts):
                    model.patches.cases_test[tick+1, nodeid] += count
                    model.patches.susceptibility_test[tick+1, nodeid] -= count


        return

    def plot(self, fig: Figure = None):
        """
        Nothing yet
        """
        return


class InfectAgentsInPatchProcess:
    """
    A component to update the infection timers of a population in a model.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize an Infect_Random_Agents instance.

        Args:

            model: The model object that contains the population.
            period: The number of ticks between each infection event.
            count: The number of agents to infect at each event.
            start (int, optional): The tick at which to start the infection events.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model object that contains the population.

        Side Effects:

        """

        self.model = model
        self.period = model.params.importation_period

        self.count = model.params.importation_count if hasattr(model.params, "importation_count") else 1
        self.patchlist = (
            model.params.importation_patchlist if hasattr(model.params, "importation_patchlist") else np.arange(model.patches.count)
        )
        self.start = model.params.importation_start if hasattr(model.params, "importation_start") else 0
        self.end = model.params.importation_end if hasattr(model.params, "importation_end") else model.params.nticks

        return

    def __call__(self, model, tick) -> None:
        """
        Updates the infection timers for the population in the model.

        Args:

            model: The model containing the population data.
            tick: The current tick or time step in the simulation.

        Returns:

            None
        """
        if (tick >= self.start) and ((tick - self.start) % self.period == 0) and (tick < self.end):
            for patch in self.patchlist:
                seed_infections_in_patch(model, patch, self.count)

        return

    def plot(self, fig: Figure = None):
        """
        Nothing yet
        """
        return
