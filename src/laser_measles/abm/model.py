"""
This module defines a `Model` class for simulating classic "generic" disease models (SI, SIS, SIR, SEIR, ...),
with options for simple demographics (births, deaths, aging) and single or multiple patches with flexible connection structure.

Classes:
    Model: A general class from which to define specific types of simulation models.

Imports:
    - datetime: For handling date and time operations.
    - click: For command-line interface utilities.
    - numpy as np: For numerical operations.
    - pandas as pd: For data manipulation and analysis.
    - laser_core.demographics: For demographic data handling.
    - laser_core.laserframe: For handling laser frame data structures.
    - laser_core.migration: For migration modeling.
    - laser_core.propertyset: For handling property sets.
    - laser_core.random: For random number generation.
    - matplotlib.pyplot as plt: For plotting.
    - matplotlib.backends.backend_pdf: For PDF generation.
    - matplotlib.figure: For figure handling.
    - tqdm: For progress bar visualization.

Model Class:
    Methods:
        __init__(self, scenario: pd.DataFrame, parameters: PropertySet, name: str) -> None:
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


import numpy as np
import polars as pl
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from laser_measles.base import BaseLaserModel
from laser_measles.utils import cast_type, StateArray

from .components.process_births import BirthsProcess
from .components.process_births_contant_pop import BirthsConstantPopProcess
from .params import ABMParams


class ABMModel(BaseLaserModel):
    """
    A class to represent the agent-based model.
    """

    def __init__(self, scenario: pl.DataFrame, parameters: ABMParams, name: str = "abm") -> None:
        """
        Initialize the disease model with the given scenario and parameters.

        Args:

            scenario (pl.DataFrame): A DataFrame containing the metapopulation patch data, including population, latitude, and longitude.
            parameters (ABMParams): A set of parameters for the model and simulations.
            name (str, optional): The name of the model. Defaults to "abm".

        Returns:

            None
        """
        super().__init__(scenario, parameters, name)

        print(f"Initializing the {name} model with {len(scenario)} patches…")

        # Setup patches and people - initialization is done via components
        self.setup_patches(scenario, parameters)
        self.setup_people(scenario, parameters)

        return

    def setup_patches(self, scenario: pl.DataFrame, parameters: PropertySet) -> None:
        self.patches = LaserFrame(capacity=len(scenario))
        # Create the state vector for each of the patches (4, num_patches) for SEIR
        self.patches.add_vector_property("states", len(self.params.states))  # S, E, I, R
        
        # Wrap the states array with StateArray for attribute access
        self.patches.states = StateArray(self.patches.states, state_names=self.params.states)

        # Start with totally susceptible population
        self.patches.states.S[:] = scenario["pop"]  # All susceptible initially
        self.patches.states.E[:] = 0  # No exposed initially
        self.patches.states.I[:] = 0  # No infected initially
        self.patches.states.R[:] = 0  # No recovered initially

        return

    def setup_people(self, scenario: pl.DataFrame, parameters: PropertySet) -> None:
        capacity = np.sum(self.patches.populations) # TODO: capacity should be set by vital dynamics
        self.people = LaserFrame(capacity=int(capacity), initial_count=0)
        self.people.add_scalar_property("nodeid", dtype=np.uint16)
        self.people.add_scalar_property("state", dtype=np.uint8, default=0)
        self.people.add_scalar_property("susceptibility", dtype=np.float32, default=0)

        for nodeid, count in enumerate(self.patches.populations):
            first, last = self.people.add(count)
            self.people.nodeid[first:last] = nodeid

        return

    def _setup_components(self) -> None:
        """
        Setup birth component registration for generic model.
        """
        births = next(filter(lambda obj: isinstance(obj, (BirthsProcess, BirthsConstantPopProcess)), self.instances), None)
        # TODO: raise an exception if there are components with an on_birth function but no Births component
        for instance in self.instances:
            if births is not None and "on_birth" in dir(instance):
                births.initializers.append(instance)

    def __call__(self, model, tick: int) -> None:
        """
        Updates the population of patches for the next tick. Copies the previous
        population data to the next tick to be updated, optionally, by a Birth and/or
        Mortality component.

        Args:

            model: The model containing the patches and their populations.
            tick (int): The current time step or tick.

        Returns:

            None
        """
        return


    def plot(self, fig: Figure = None):
        """
        Plots various visualizations related to the scenario and population data.

        Parameters:

            fig (Figure, optional): A matplotlib Figure object to use for plotting. If None, a new figure will be created.

        Yields:

            None: This function uses a generator to yield control back to the caller after each plot is created.

        The function generates three plots:

            1. A scatter plot of the scenario patches and populations.
            2. A histogram of the distribution of the day of birth for the initial population.
            3. A pie chart showing the distribution of update phase times.
        """

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Scenario Patches and Populations")
        if "geometry" in self.scenario.columns:
            ax = plt.gca()
            self.scenario.plot(ax=ax)
        scatter = plt.scatter(
            self.scenario.longitude,
            self.scenario.latitude,
            s=self.scenario.population / 1000,
            c=self.scenario.population,
            cmap="inferno",
        )
        plt.colorbar(scatter, label="Population")

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Distribution of Day of Birth for Initial Population")

        count = self.patches.populations[0, :].sum()  # just the initial population
        dobs = self.people.dob[0:count]
        plt.hist(dobs, bins=100)
        plt.xlabel("Day of Birth")

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        metrics = pl.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()

        plt.pie(
            sum_columns,
            labels=[name if not name.startswith("do_") else name[3:] for name in sum_columns.index],
            autopct="%1.1f%%",
            startangle=140,
        )
        plt.title("Update Phase Times")

        yield
        return

# Alias for backwards compatibility
Model = ABMModel