"""
This module defines the Births component, which is responsible for simulating births in a population model.

Classes:

    Births:

        Manages the birth process within a population model, including initializing births, updating population data, and plotting birth statistics.

Usage:

    The Births component requires a model with a `population` attribute that has a `dob` attribute.
    It calculates the number of births based on the model's parameters and updates the population
    accordingly. It also provides methods to plot birth statistics.

Example:

    model = YourModelClass()
    births = Births(model)
    births(model, tick)
    births.plot()

Attributes:

    model (object): The population model.
    _initializers (list): List of initializers to be called on birth.
    _metrics (list): List to store timing metrics for initializers.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Births:
    """
    A component to handle the birth events in a model.

    Attributes:

        model: The model instance containing population and parameters.
        verbose (bool): Flag to enable verbose output. Default is False.
        initializers (list): List of initializers to be called on birth events.
        metrics (DataFrame): DataFrame to holding timing metrics for initializers.
    """

    def __init__(self, model, verbose: bool = False):
        """
        Initialize the Births component.

        Parameters:

            model (object): The model object which must have a `population` attribute.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Raises:

            AssertionError: If the model does not have a `population` attribute.
            AssertionError: If the model's population does not have a `dob` attribute.
        """

        assert getattr(model, "population", None) is not None, "Births requires the model to have a `population` attribute"
        assert getattr(model.population, "dob", None) is not None, "Births requires the model population to have a `dob` attribute"

        self.model = model

        nyears = (model.params.nticks + 364) // 365
        model.patches.add_vector_property("births", length=nyears, dtype=np.uint32)

        self._initializers = []
        self._metrics = []

        return

    @property
    def initializers(self):
        """
        Returns the initializers to call on new agent births.

        This method retrieves the initializers that are used to set up the
        initial state or configuration for agents at birth.

        Returns:

            list: A list of initializers - instances of objects with an `on_birth` method.
        """

        return self._initializers

    @property
    def metrics(self):
        """
        Returns the timing metrics for the births initializers.

        This method retrieves the timing metrics for the births initializers.

        Returns:

            DataFrame: A Pandas DataFrame of timing metrics for the births initializers.
        """

        return pd.DataFrame(self._metrics, columns=["tick"] + [type(initializer).__name__ for initializer in self._initializers])

    def __call__(self, model, tick) -> None:
        """
        Adds new agents to each patch based on expected daily births calculated from CBR. Calls each of the registered initializers for the newborns.

        Args:

            model: The simulation model containing patches, population, and parameters.
            tick: The current time step in the simulation.

        Returns:

            None

        This method performs the following steps:

            1. Calculates the day of the year (doy) and the current year based on the tick.
            2. On the first day of the year, it generates annual births for each patch using a Poisson distribution.
            3. Calculates the number of births for the current day.
            4. Adds the newborns to the population and sets their date of birth.
            5. Assigns node IDs to the newborns.
            6. Calls any additional initializers for the newborns and records the timing of these initializations.
            7. Updates the population counts for the next tick with the new births.
        """

        doy = tick % 365 + 1  # day of year 1…365
        year = tick // 365

        if doy == 1:
            model.patches.births[year, :] = model.prng.poisson(model.patches.populations[tick, :] * model.params.cbr / 1000)

        annual_births = model.patches.births[year, :]
        todays_births = (annual_births * doy // 365) - (annual_births * (doy - 1) // 365)
        count_births = todays_births.sum()
        istart, iend = model.population.add(count_births)

        model.population.dob[istart:iend] = tick  # set to current tick

        # set the nodeids for the newborns in case subsequent initializers need them (properties varying by patch)
        index = istart
        nodeids = model.population.nodeid
        for nodeid, births in enumerate(todays_births):
            nodeids[index : index + births] = nodeid
            index += births

        timing = [tick]
        for initializer in self._initializers:
            tstart = datetime.now(tz=None)  # noqa: DTZ005
            initializer.on_birth(model, tick, istart, iend)
            tfinish = datetime.now(tz=None)  # noqa: DTZ005
            delta = tfinish - tstart
            timing.append(int(delta.total_seconds() * 1_000_000))
        self._metrics.append(timing)

        model.patches.populations[tick + 1, :] += todays_births

        return

    def plot(self, fig: Figure = None):
        """
        Plots the births in the top 5 most populous patches and a pie chart of birth initializer times.

        Parameters:

            fig (Figure, optional): A matplotlib Figure object. If None, a new figure will be created. Defaults to None.

        Yields:

            None: This function yields twice to allow for intermediate plotting steps.
        """

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Births in Top 5 Most Populous Patches")

        indices = self.model.patches.populations[0, :].argsort()[-5:]
        ax1 = plt.gca()
        ticks = list(range(0, self.model.params.nticks, 365))
        for index in indices:
            ax1.plot(self.model.patches.populations[ticks, index], marker="x", markersize=4)

        ax2 = ax1.twinx()
        for index in indices:
            ax2.plot(self.model.patches.births[:, index], marker="+", markersize=4)

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Births in Top 5 Most Populous Patches")

        metrics = pd.DataFrame(self._metrics, columns=["tick"] + [type(initializer).__name__ for initializer in self._initializers])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()

        plt.pie(sum_columns, labels=sum_columns.index, autopct="%1.1f%%", startangle=140)
        plt.title("On Birth Initializer Times")

        yield

        return
