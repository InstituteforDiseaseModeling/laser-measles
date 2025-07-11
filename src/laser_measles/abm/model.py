"""
A class to represent the agent-based model.
"""

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from laser_measles.abm.base import BaseABMScenario
from laser_measles.abm.base import PatchLaserFrame
from laser_measles.abm.base import PeopleLaserFrame
from laser_measles.base import BaseLaserModel, BaseScenario
from laser_measles.utils import StateArray

from . import components
from .params import ABMParams


class ABMModel(BaseLaserModel):
    """
    A class to represent the agent-based model.
    """

    people: PeopleLaserFrame

    # Specify the scenario wrapper class for auto-wrapping DataFrames
    scenario_wrapper_class = BaseABMScenario

    def __init__(self, scenario: BaseABMScenario | pl.DataFrame, params: ABMParams, name: str = "abm") -> None:
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

    def __call__(self, model, tick: int) -> None:
        pass

    def setup_patches(self) -> None:
        """Setup the patches for the model."""

        scenario: BaseScenario = self.scenario

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

        return

    def setup_people(self) -> None:
        """Placeholder for people - sets the data types for patch_id and susceptibility."""

        self.people = PeopleLaserFrame(capacity=1)
        self.people.add_scalar_property("patch_id", dtype=np.uint16)  # patch id
        self.people.add_scalar_property("state", dtype=np.uint8, default=0)  # state
        self.people.add_scalar_property("susceptibility", dtype=np.float32, default=0)  # susceptibility factor

        return

    def initialize_people_capacity(self, capacity: int, initial_count: int = -1) -> None:
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
        new_people = type(self.people).create_with_capacity(capacity, self.people, initial_count=initial_count)

        # Update the people laserframe
        self.people = new_people

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

    def _setup_components(self) -> None:
        pass

    def _initialize(self) -> None:
        """
        Setup birth component registration for generic model.
        """

        # This will re-run all instantiaion
        if len(self.people) != self.patches.states.sum():
            if self.params.verbose:
                print("No vital dynamics provided. Creating a new people laserframe with the same properties as the patches.")
            self.prepend_component(components.NoBirthsProcess)

        super()._initialize()

        return


# Alias for backwards compatibility
Model = ABMModel
