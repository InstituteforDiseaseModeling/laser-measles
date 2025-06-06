"""
This module defines the `Model` class for simulation

Classes:
    Model: A class to represent the simulation model.

Imports:


Model Class:
    Methods:
        __init__(self, scenario: pd.DataFrame, parameters: PropertySet, name: str = "template") -> None:
            Initializes the model with the given scenario and parameters.

        components(self) -> list:
            Gets the list of components in the model.

        components(self, components: list) -> None:
            Sets the list of components in the model and initializes instances and phases.

        __call__(self, model, tick: int) -> None:
            Updates the model for a given tick.

        run(self) -> None:
            Runs the model for the specified number of ticks.

        visualize(self, pdf: bool = True) -> None:
            Generates visualizations of the model's results, either displaying them or saving to a PDF.

        plot(self, fig: Figure = None):
            Generates plots for the scenario patches and populations, distribution of day of birth, and update phase times.
"""

from datetime import datetime

import alive_progress
import click
import polars as pl
from laser_core.laserframe import LaserFrame
from laser_core.random import seed as seed_prng

from laser_measles.biweekly.base import BaseScenario
from laser_measles.biweekly.components import Infection
from laser_measles.biweekly.components import VitalDynamics
from laser_measles.biweekly.params import BiweeklyParams


class BiweeklyModel:
    """
    A class to represent the biweekly model.

    Args:

        scenario (pl.DataFrame): A DataFrame containing the scenario data, including population, latitude, and longitude.
        parameters (BiweeklyParams): A set of parameters for the model.
        name (str, optional): The name of the model. Defaults to "template".

    Notes:

        This class initializes the model with the given scenario and parameters. The scenario DataFrame must include the following columns:

            - `ids` (string): The name of the patch or location.
            - `pop` (integer): The population count for the patch.
            - `lat` (float degrees): The latitude of the patches (e.g., from geographic or population centroid).
            - `lon` (float degrees): The longitude of the patches (e.g., from geographic or population centroid).
            - `mcv1` (float): The MCV1 coverage for the patches.
    """

    def __init__(self, scenario: BaseScenario, parameters: BiweeklyParams, name: str = "biweekly") -> None:
        """
        Initialize the disease model with the given scenario and parameters.

        Args:

            scenario (BaseScenario): A DataFrame containing the scenario data, including population, latitude, and longitude.
            parameters (PropertySet): A set of parameters for the model, including seed, nticks, k, a, b, c, max_frac, cbr, verbose, and pyramid_file.
            name (str, optional): The name of the model. Defaults to "biweekly".

        Returns:

            None
        """

        self.tinit = datetime.now(tz=None)  # noqa: DTZ005
        self.scenario = scenario
        self.params = parameters
        self.name = name

        # seed the random number generator
        self.prng = seed_prng(parameters.seed if parameters.seed is not None else self.tinit.microsecond)

        # Add nodes to the model
        num_nodes = len(scenario)
        self.nodes = LaserFrame(num_nodes)

        # Initialize time tracking from parameters
        self.start_time = datetime.strptime(self.params.start_time, "%Y-%m").replace(tzinfo=datetime.now().astimezone().tzinfo)
        self.current_date = self.start_time

        # create the state vector for each of the nodes (3, num_nodes)
        self.nodes.add_vector_property("states", len(self.params.states))  # S, I, R

        self.components = [Infection, VitalDynamics]

        return

    @property
    def components(self) -> list:
        """
        Retrieve the list of model components.

        Returns:

            list: A list containing the components.
        """

        return self._components

    @components.setter
    def components(self, components: list) -> None:
        """
        Sets up the components of the model and initializes instances and phases.

        This function takes a list of component types, creates an instance of each, and adds each callable component to the phase list.

        Args:

            components (list): A list of component classes to be initialized and integrated into the model.

        Returns:

            None
        """

        self._components = components
        self.instances = [self]  # instantiated instances of components
        self.phases = [self]  # callable phases of the model
        for component in components:
            instance = component(self, self.params.verbose)
            self.instances.append(instance)
            if "__call__" in dir(instance):
                self.phases.append(instance)

        return

    def __call__(self, model, tick: int) -> None:
        """
        Updates the model for the next tick.

        Args:

            model: The model containing the patches and their populations.
            tick (int): The current time step or tick.

        Returns:

            None
        """

        return

    def step(self, tick) -> None:
        timing = [tick]
        for phase in self.phases:
            tstart = datetime.now(tz=None)  # noqa: DTZ005
            phase(self, tick)
            tfinish = datetime.now(tz=None)  # noqa: DTZ005
            delta = tfinish - tstart
            timing.append(delta.seconds * 1_000_000 + delta.microseconds)
        self.metrics.append(timing)

    def run(self) -> None:
        """
        Execute the model for a specified number of ticks, recording the time taken for each phase.

        This method initializes the start time, iterates over the number of ticks specified in the model parameters,
        and for each tick, it executes each phase of the model while recording the time taken for each phase.

        The metrics for each tick are stored in a list. After completing all ticks, it records the finish time and,
        if verbose mode is enabled, prints a summary of the timing metrics.

        Attributes:

            tstart (datetime): The start time of the model execution.
            tfinish (datetime): The finish time of the model execution.
            metrics (list): A list of timing metrics for each tick and phase.

        Returns:

            None
        """

        self.tstart = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tstart}: Running the {self.name} model for {self.params.nticks} ticks…")

        self.metrics = []
        with alive_progress.alive_bar(total=self.params.nticks) as bar:
            for tick in range(self.params.nticks):
                self.step(tick)
                bar()
        self.tfinish = datetime.now(tz=None)  # noqa: DTZ005
        print(f"Completed the {self.name} model at {self.tfinish}…")

        if self.params.verbose:
            metrics = pl.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
            plot_columns = metrics.columns[1:]
            sum_columns = metrics[plot_columns].sum()
            width = max(map(len, sum_columns.index))
            for key in sum_columns.index:
                print(f"{key:{width}}: {sum_columns[key]:13,} µs")
            print("=" * (width + 2 + 13 + 3))
            print(f"{'Total:':{width + 1}} {sum_columns.sum():13,} microseconds")

        return
