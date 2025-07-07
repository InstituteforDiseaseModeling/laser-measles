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
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from laser_measles.abm.base import BaseABMScenario
from laser_measles.abm.base import PatchLaserFrame
from laser_measles.abm.base import PeopleLaserFrame
from laser_measles.base import BaseLaserModel
from laser_measles.utils import StateArray
from laser_measles.utils import get_laserframe_properties

from . import components
from .params import ABMParams


class ABMModel(BaseLaserModel[BaseABMScenario, ABMParams]):
    """
    A class to represent the agent-based model.
    """

    people: PeopleLaserFrame

    def __init__(self, scenario: BaseABMScenario, params: ABMParams, name: str = "abm") -> None:
        """
        Initialize the disease model with the given scenario and parameters.

        Args:

            scenario (pl.DataFrame): A DataFrame containing the metapopulation patch data, including population, latitude, and longitude.
            parameters (ABMParams): A set of parameters for the model and simulations.
            name (str, optional): The name of the model. Defaults to "abm".

        Returns:

            None
        """
        super().__init__(scenario, params, name)

        if self.params.verbose:
            print(f"Initializing the {name} model with {len(scenario)} patches…")

        # Setup patches
        self.setup_patches()
        # Setup people - initialization is done via components
        self.setup_people()

        return

    def setup_patches(self) -> None:
        """Setup the patches for the model."""

        scenario = self.scenario

        self.patches = PatchLaserFrame(capacity=len(scenario))
        # Create the state vector for each of the patches (4, num_patches) for SEIR
        self.patches.add_vector_property("states", len(self.params.states))  # S, E, I, R

        # Wrap the states array with StateArray for attribute access
        self.patches.states = StateArray(self.patches.states, state_names=self.params.states)

        # Start with totally susceptible population
        self.patches.states.S[:] = scenario["pop"]  # All susceptible initially
        self.patches.states.E[:] = 0  # No exposed initially
        self.patches.states.I[:] = 0  # No infected initially
        self.patches.states.R[:] = 0  # No recovered initially
        self.patches.states.D[:] = 0  # No deaths initially

        return

    def setup_people(self) -> None:
        """Placeholder for people - sets the data types for patch_id and susceptibility."""

        self.people = PeopleLaserFrame(capacity=1)
        self.people.add_scalar_property("patch_id", dtype=np.uint16)  # patch id
        self.people.add_scalar_property("state", dtype=np.uint8, default=0)  # state
        self.people.add_scalar_property("susceptibility", dtype=np.float32, default=0)  # susceptibility factor

        return

    def initialize_people_capacity(self, capacity: int) -> None:
        """
        Initialize the people LaserFrame with a new capacity while preserving all properties.

        This method uses the factory method from BasePeopleLaserFrame to create a new
        instance of the same type with the specified capacity, copying all properties
        from the existing instance.

        Args:
            capacity: The new capacity for the people LaserFrame
        """
        if self.people is None:
            raise RuntimeError("Cannot initialize capacity: people LaserFrame is None")

        # Use the factory method to create a new instance with the same type and properties
        new_people = type(self.people).create_with_capacity(capacity, self.people)

        # Update the people laserframe
        self.people = new_people

    def initialize(self) -> None:
        """
        Setup birth component registration for generic model.
        """

        # This will re-run all instantiaion
        if len(self.people) != self.patches.states.sum():
            if self.params.verbose:
                print("No vital dynamics provided. Creating a new people laserframe with the same properties as the patches.")
            self.prepend_component(components.NoBirthsProcess)

        super().initialize()

    def __call__(self, model, tick: int) -> None:
        return

    def plot(self, fig: Figure | None = None):
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
            self.scenario.lat,
            self.scenario.lon,
            s=self.scenario.pop / 1000,
            c=self.scenario.pop,
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
        column_names = ["tick"] + [type(phase).__name__ for phase in self.phases]
        metrics = pl.DataFrame(self.metrics, schema=column_names)
        sum_columns = metrics.select([pl.sum(col).alias(col) for col in metrics.columns[1:]]).to_dict(as_series=False)

        # Build labels (strip "do_" if present)
        labels = [name[3:] if name.startswith("do_") else name for name in sum_columns.keys()]
        values = list(sum_columns.values())

        # Plot pie chart
        plt.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
        )
        plt.title("Update Phase Times")

        yield
        return


# Alias for backwards compatibility
Model = ABMModel
