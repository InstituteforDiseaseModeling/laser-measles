"""
This module defines the NonDiseaseDeaths class, which models non-disease related deaths in a population over time.

Classes:
    NonDiseaseDeaths: A class to handle non-disease mortality in a population model.

Dependencies:
    - click
    - numpy as np
    - laser_core.demographics.KaplanMeierEstimator
    - laser_core.sortedqueue.SortedQueue
    - matplotlib.pyplot as plt
    - matplotlib.figure.Figure
    - alive_progress

Usage:
    The NonDiseaseDeaths class is initialized with a model and an optional verbosity flag. It adds scalar properties to the population,
    estimates dates of death using a Kaplan-Meier estimator, and manages a sorted queue of non-disease death events. The class provides
    methods to handle births, update the model at each tick, and plot cumulative non-disease deaths.

Example:
    model = SomeModel()
    non_disease_deaths = NonDiseaseDeaths(model)
    non_disease_deaths.on_birth(model, tick, istart, iend)
    non_disease_deaths(model, tick)
    non_disease_deaths.plot()
"""

import click
import numpy as np
from alive_progress import alive_bar
from laser_core.demographics import KaplanMeierEstimator
from laser_core.sortedqueue import SortedQueue
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class NonDiseaseDeaths:
    """
    A component to model non-disease related deaths in a population.
    """

    def __init__(self, model, verbose: bool = False):
        """
        Initialize the non-disease deaths component of the model.

        Parameters:

            model : object

                The model object that contains the population and parameters.

            verbose : bool, optional

                If True, enables verbose output (default is False).

        Notes:

            - Adds scalar properties "alive" and "dod" to the model's population.
            - Initializes the Kaplan-Meier estimator from `model.params.mortality_file` with the cumulative deaths data.
            - Adds vector property "deaths" to the model's patches.
        """

        self.model = model

        model.population.add_scalar_property("alive", dtype=bool, default=True)
        # model.population.alive[0 : model.population.count] = True

        model.population.add_scalar_property("dod", dtype=np.uint16)  # Up to 65535 days in the future
        cumulative_deaths = np.loadtxt(model.params.mortality_file)
        model.estimator = KaplanMeierEstimator(cumulative_deaths)
        dods = model.population.dod[0 : model.population.count]
        dobs = model.population.dob[0 : model.population.count]
        # Use -dobs to get the current age of the agent (in days)
        dods[:] = model.estimator.predict_age_at_death(-dobs, max_year=100)

        dods -= dobs.astype(dods.dtype)  # renormalize to be relative to _now_ (t = 0)

        # add non-disease mortality to the model
        model.nddq = SortedQueue(model.population.capacity, model.population.dod)
        print("Adding agents to the non-disease death queue…")
        with alive_bar(len(np.nonzero(dods[0 : model.population.count] < model.params.nticks)[0])) as bar:
            for i in np.nonzero(dods[0 : model.population.count] < model.params.nticks)[0]:
                model.nddq.push(i)
                bar()

        # +364 to account for something other than integral numbers of years (in nticks)
        # model.patches.add_vector_property("deaths", (model.params.nticks + 364) // 365)
        model.patches.add_vector_property("deaths", length=model.params.nticks)

        return

    def on_birth(self, model, tick, istart, iend):
        """
        Handles the birth of new agents in the model.

        This function updates the population's alive status and predicted date of death (dod) for newly born agents.
        It also pushes agents with a date of death within the simulation's maximum ticks to the non-disease death queue (nddq).

        Parameters:

            model (object): The simulation model containing population and parameters.
            tick (int): The current tick or time step in the simulation.
            istart (int): The starting index of the newly born agents in the population array.
            iend (int): The ending index of the newly born agents in the population array.

        Returns:

            None
        """

        # newborns are alive and have a predicted date of death
        model.population.alive[istart:iend] = True
        model.population.dod[istart:iend] = 0  # temporarily set to 0 for the next line
        model.population.dod[istart:iend] = tick + model.estimator.predict_age_at_death(model.population.dod[istart:iend], max_year=100)

        max_tick = model.params.nticks
        dods = model.population.dod[0 : model.population.count]
        q = model.nddq
        for agent in range(istart, iend):
            if dods[agent] < max_tick:
                q.push(agent)

        return

    def __call__(self, model, tick):
        """
        Update the model state for the given tick by processing the non-disease deaths queue.

        This method updates the population and deaths counts for each node in the model
        based on the non-disease deaths queue. It marks agents as dead and updates the
        corresponding node's population and death counts.

        Parameters:

            model (object): The model object containing population and patches data.
            tick (int): The current time step or tick in the simulation.

        Returns:

            None
        """

        nodeids = model.population.nodeid[0 : model.population.count]
        node_population = model.patches.populations[tick, :]
        node_deaths = model.patches.deaths[tick, :]
        alive = model.population.alive[0 : model.population.count]
        # TODO - support registration of "on_death" callbacks for agents
        # susceptibility = model.population.susceptibility[0 : model.population.count]
        # ma_timers = model.population.ma_timers[0 : model.population.count]
        # ri_timers = model.population.ri_timers[0 : model.population.count]
        # etimers = model.population.etimers[0 : model.population.count]
        # itimers = model.population.itimers[0 : model.population.count]

        pq = model.nddq
        while (len(pq) > 0) and (pq.peekv() <= tick):
            iagent = pq.popi()
            nodeid = nodeids[iagent]
            node_population[nodeid] -= 1
            node_deaths[nodeid] += 1
            alive[iagent] = False
            # susceptibility[iagent] = 0
            # ma_timers[iagent] = 0
            # ri_timers[iagent] = 0
            # etimers[iagent] = 0
            # itimers[iagent] = 0

        return

    def plot(self, fig: Figure = None):
        """
        Plots the cumulative non-disease deaths for the year 0 population.

        Parameters:

            fig (Figure, optional): A matplotlib Figure object. If None, a new figure is created. Defaults to None.

        Returns:

            None

        Yields:

            None

        Notes:

        - The function plots two lines:

            1. The cumulative number of non-disease deaths over the years since birth, marked with red 'x'.
            2. The expected cumulative deaths based on the model's estimator, marked with blue '+'.

        - If no individuals are found born in the first year, a message is printed.
        """

        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Cumulative Non-Disease Deaths for Year 0 Population")

        dobs = self.model.population.dob[0 : self.model.population.count]
        dods = self.model.population.dod[0 : self.model.population.count]
        individuals = np.nonzero((0 < dobs) & (dobs < 365))[0]
        if len(individuals) > 0:
            ages_at_death = (dods[individuals] - dobs[individuals]) // 365
            aad_max = ages_at_death.max()
            counts = np.zeros(aad_max + 1, dtype=np.int32)
            np.add.at(counts, ages_at_death, 1)  # [individuals], 1)
            cumulative = counts.cumsum()
            plt.plot(range(aad_max + 1), cumulative, marker="x", markersize=4, color="red")

            percentage = self.model.estimator.cumulative_deaths / self.model.estimator.cumulative_deaths[-1]
            expected_deaths = np.round(len(individuals) * percentage).astype(np.uint32)

            plt.plot(range(aad_max + 1), expected_deaths, marker="+", markersize=4, color="blue")
            plt.xlabel("Years Since Birth")
            yield
        else:
            click.echo("Found no individuals born in the first year.")

        return
