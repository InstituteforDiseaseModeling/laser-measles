"""
Base classes for laser-measles components and models

This module contains the base classes for laser-measles components and models.

The BaseComponent class is the base class for all laser-measles components.
It provides a uniform interface for all components with a __call__(model, tick) method
for execution during simulation loops.

The BaseLaserModel class is the base class for all laser-measles models.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Generic
from typing import Protocol
from typing import TypeVar

import alive_progress
import matplotlib.pyplot as plt
import polars as pl
from laser_core.laserframe import LaserFrame
from laser_core.random import seed as seed_prng
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from laser_measles.utils import StateArray
from laser_measles.wrapper import pretty_laserframe

ScenarioType = TypeVar("ScenarioType")
ModelType = TypeVar("ModelType")


class ParamsProtocol(Protocol):
    """Protocol defining the expected structure of model parameters."""

    seed: int
    start_time: str
    num_ticks: int
    verbose: bool

    @property
    def time_step_days(self) -> int: ...
    @property
    def states(self) -> list[str]: ...


ParamsType = TypeVar("ParamsType", bound=ParamsProtocol)


@pretty_laserframe
class BasePatchLaserFrame(LaserFrame):
    """LaserFrame that has a states property."""

    states: StateArray  # StateArray with attribute access (S, E, I, R, etc.)


@pretty_laserframe
class BasePeopleLaserFrame(LaserFrame):
    """
    Base class for people LaserFrames with enhanced printing capabilities.

    This class provides factory methods for creating new instances with the same
    properties but different capacity, making it easy to resize people collections.
    """

    @classmethod
    def create_with_capacity(cls, capacity: int, source_frame: BasePeopleLaserFrame) -> Any:
        """
        Create a new instance of the same type with specified capacity.

        This factory method creates a new instance of the same class as the source_frame,
        with the specified capacity, and copies all properties from the source.

        Args:
            capacity: The capacity for the new LaserFrame
            source_frame: The source LaserFrame to copy properties from

        Returns:
            A new instance of the same type with copied properties
        """
        # Create new instance of the same type
        new_frame = cls(capacity=capacity)

        # Copy all properties from source
        new_frame.copy_properties_from(source_frame)

        return new_frame
        """
        Create a new instance of the same type with specified capacity.
        
        This factory method creates a new instance of the same class as the source_frame,
        with the specified capacity, and copies all properties from the source.
        
        Args:
            capacity: The capacity for the new LaserFrame
            source_frame: The source LaserFrame to copy properties from
            
        Returns:
            A new instance of the same type with copied properties
        """
        # Create new instance of the same type
        new_frame = cls(capacity=capacity)

        # Copy all properties from source
        new_frame.copy_properties_from(source_frame)

        return new_frame

    def copy_properties_from(self, source_frame: BasePeopleLaserFrame) -> None:
        """
        Copy all properties from another LaserFrame instance.

        This method copies all scalar and vector properties from the source frame,
        including their data types and default values.

        Args:
            source_frame: The source LaserFrame to copy properties from
        """
        from laser_measles.utils import get_laserframe_properties

        properties = get_laserframe_properties(source_frame)

        for property_name in properties:
            source_property = getattr(source_frame, property_name)

            if source_property.ndim == 1:
                # Scalar property
                self.add_scalar_property(
                    property_name, dtype=source_property.dtype, default=source_property[0] if len(source_property) > 0 else 0
                )
            elif source_property.ndim == 2:
                # Vector property
                self.add_vector_property(
                    property_name,
                    len(source_property),
                    dtype=source_property.dtype,
                    default=source_property[:, 0] if source_property.shape[1] > 0 else 0,
                )
            else:
                # Handle higher dimensional properties if needed
                raise NotImplementedError(f"Property {property_name} has {source_property.ndim} dimensions, not supported")


class BaseLaserModel(ABC, Generic[ScenarioType, ParamsType]):
    """
    Base class for laser-measles simulation models.

    Provides common functionality for model initialization, component management,
    timing, metrics collection, and execution loops.
    """

    # Type annotations for attributes that subclasses will set
    patches: BasePatchLaserFrame

    def __init__(self, scenario: ScenarioType, params: ParamsType, name: str) -> None:
        """
        Initialize the model with common attributes.

        Args:
            scenario: Scenario data (type varies by model)
            params: Model parameters (type varies by model)
            name: Model name
        """
        self.tinit = datetime.now(tz=None)  # noqa: DTZ005
        if params.verbose:
            print(f"{self.tinit}: Creating the {name} model…")

        self.scenario = scenario
        self.params = params
        self.name = name

        # Initialize random number generator
        seed_value = params.seed if hasattr(params, "seed") and params.seed is not None else self.tinit.microsecond
        self.prng = seed_prng(seed_value)

        # Component management attributes
        self._components: list = []
        self.instances: list = []
        self.phases: list = []  # Called every tick

        # Metrics and timing
        self.metrics: list = []
        self.tstart: datetime | None = None
        self.tfinish: datetime | None = None

        # Time tracking
        self.start_time = datetime.strptime(self.params.start_time, "%Y-%m")  # noqa DTZ007
        self.current_date = self.start_time

    @property
    def components(self) -> list:
        """
        Retrieve the list of model components.

        Returns:
            list: A list containing the components.
        """
        return self._components

    @components.setter
    def components(self, components: list[type[BaseComponent]]) -> None:
        """
        Sets up the components of the model and constructs all instances.

        Args:
            components (list): A list of component classes to be initialized and integrated into the model.
        """
        self._components = components
        self.instances = []
        self.phases = []
        for component in components:
            instance = component(self, verbose=getattr(self.params, "verbose", False))
            self.instances.append(instance)
            if "__call__" in dir(instance):
                self.phases.append(instance)

        # Allow subclasses to perform additional component setup
        self._setup_components()

    def add_component(self, component: type[BaseComponent]) -> None:
        """
        Add the component class and an instance in model.instances. Note that this does not create new instances of othr components.

        Args:
            component (BaseComponent): A component class to be initialized and integrated into the model.
        """
        self._components.append(component)
        instance = component(self, verbose=getattr(self.params, "verbose", False))
        self.instances.append(instance)
        if "__call__" in dir(instance):
            self.phases.append(instance)
        self._setup_components()

    def prepend_component(self, component: type[BaseComponent]) -> None:
        self._components.insert(0, component)
        instance = component(self, verbose=getattr(self.params, "verbose", False))
        self.instances.insert(0, instance)
        if "__call__" in dir(instance):
            self.phases.insert(0, instance)
        self._setup_components()

    def _setup_components(self) -> None:
        """
        Hook for subclasses to perform additional component setup.
        Override in subclasses as needed.
        """

    @abstractmethod
    def __call__(self, model: Any, tick: int) -> None:
        """
        Updates the model for a given tick.

        Args:
            model (BaseLaserModel): The model instance
            tick (int): The current time step or tick
        """

    def run(self) -> None:
        """
        Execute the model for a specified number of ticks, recording timing metrics.
        """
        # Check that there are some components to the model
        if len(self.components) == 0:
            raise RuntimeError("No components have been added to the model")

        # Initialize all component instances
        self.initialize()

        # TODO: Check that the model has been initialized
        num_ticks = self.params.num_ticks
        self.tstart = datetime.now(tz=None)  # noqa: DTZ005
        if self.params.verbose:
            print(f"{self.tstart}: Running the {self.name} model for {num_ticks} ticks…")

        self.metrics = []
        with alive_progress.alive_bar(num_ticks) as bar:
            for tick in range(num_ticks):
                self._execute_tick(tick)
                bar()

        self.tfinish = datetime.now(tz=None)  # noqa: DTZ005
        if self.params.verbose:
            print(f"Completed the {self.name} model at {self.tfinish}…")
            self._print_timing_summary()

    def _execute_tick(self, tick: int) -> None:
        """
        Execute a single tick. Can be overridden by subclasses for custom behavior.

        Args:
            tick: The current tick number
        """
        timing = [tick]
        for phase in self.phases:
            tstart = datetime.now(tz=None)  # noqa: DTZ005
            phase(self, tick)
            tfinish = datetime.now(tz=None)  # noqa: DTZ005
            delta = tfinish - tstart
            timing.append(delta.seconds * 1_000_000 + delta.microseconds)
        self.metrics.append(timing)

        # Update current date by time_step_days
        self.current_date += timedelta(days=self.params.time_step_days)

    def time_elapsed(self, units: str = "days") -> int:
        """
        Return time elapsed since the start of the model.
        """
        if units == "days":
            return (self.current_date - self.start_time).days
        else:
            raise ValueError(f"Invalid time units: {units}")

    def initialize(self) -> None:
        """
        Initialize all component instances in the model.

        This method calls initialize() on all component instances and sets
        their initialized flag to True after successful initialization.
        """
        for instance in self.instances:
            if hasattr(instance, "initialize") and hasattr(instance, "initialized"):
                instance.initialize(self)
                instance.initialized = True

    def _print_timing_summary(self) -> None:
        """
        Print timing summary for verbose mode.
        """
        try:
            import pandas as pd  # noqa: PLC0415

            names = [type(phase).__name__ for phase in self.phases]
            metrics = pd.DataFrame(self.metrics, columns=["tick"] + names)
            plot_columns = metrics.columns[1:]
            sum_columns = metrics[plot_columns].sum()
            width = max(map(len, sum_columns.index))
            for key in sum_columns.index:
                print(f"{key:{width}}: {sum_columns[key]:13,} µs")
            print("=" * (width + 2 + 13 + 3))
            print(f"{'Total:':{width + 1}} {sum_columns.sum():13,} microseconds")
        except ImportError:
            try:
                import polars as pl  # noqa: PLC0415

                names = [type(phase).__name__ for phase in self.phases]
                metrics = pl.DataFrame(self.metrics, schema=["tick"] + names)
                plot_columns = metrics.columns[1:]
                sum_columns = metrics.select(plot_columns).sum()
                # Handle polars DataFrame differently
                print("Timing summary available but detailed formatting requires pandas")
            except ImportError:
                print("Timing summary requires pandas or polars")

    def cleanup(self) -> None:
        """
        Clean up model resources to prevent memory leaks.

        This method should be called when the model is no longer needed
        to free up memory from LaserFrame objects and other large data structures.
        """
        try:
            # Clear LaserFrame objects
            if hasattr(self, "patches") and self.patches is not None:
                # Clear all properties from the LaserFrame
                if hasattr(self.patches, "_properties"):
                    for prop_name in list(self.patches._properties.keys()):
                        setattr(self.patches, prop_name, None)
                    self.patches._properties.clear()

                # Reset LaserFrame capacity and count
                if hasattr(self.patches, "_capacity"):
                    self.patches._capacity = 0
                if hasattr(self.patches, "_count"):
                    self.patches._count = 0

                self.patches = None

            if hasattr(self, "people") and self.people is not None:
                # Clear all properties from the LaserFrame
                if hasattr(self.people, "_properties"):
                    for prop_name in list(self.people._properties.keys()):
                        setattr(self.people, prop_name, None)
                    self.people._properties.clear()

                # Reset LaserFrame capacity and count
                if hasattr(self.people, "_capacity"):
                    self.people._capacity = 0
                if hasattr(self.people, "_count"):
                    self.people._count = 0

                self.people = None

            # Clear component instances and their references
            if hasattr(self, "instances"):
                for instance in self.instances:
                    # Clear any LaserFrame references in components
                    if hasattr(instance, "model"):
                        instance.model = None
                    # Clear any large data structures in components
                    for attr_name in dir(instance):
                        if not attr_name.startswith("_") and attr_name not in ["initialized", "verbose"]:
                            attr_value = getattr(instance, attr_name, None)
                            if hasattr(attr_value, "__len__") and not callable(attr_value):
                                try:
                                    setattr(instance, attr_name, None)
                                except (AttributeError, TypeError):
                                    pass  # Skip if attribute is read-only
                self.instances.clear()

            # Clear phases and components
            if hasattr(self, "phases"):
                self.phases.clear()
            if hasattr(self, "_components"):
                self._components.clear()

            # Clear metrics and other large data structures
            if hasattr(self, "metrics"):
                self.metrics.clear()

            # Clear scenario and params references to large data
            if hasattr(self, "scenario"):
                del self.scenario
            if hasattr(self, "params"):
                # Clear any large data structures in params
                del self.params

            # Clear random number generator
            if hasattr(self, "prng"):
                del self.prng

        except Exception as e:
            # Don't let cleanup errors crash the program
            print(f"Warning: Error during model cleanup: {e}")

    def get_instance(self, cls: type | str) -> list:
        """
        Get all instances of a specific component class.

        Args:
            cls: The component class to search for

        Returns:
            List of instances of the specified class, or None if none found.
            Works with inheritance - subclasses will match parent class searches.

        Example:
            state_trackers = model.get_instance(StateTracker)
            if state_trackers:
                state_tracker = state_trackers[0]  # Get first instance
        """
        if isinstance(cls, str):
            matches = [instance for instance in self.instances if instance.name == cls]
        else:
            matches = [instance for instance in self.instances if isinstance(instance, cls)]
        return matches if matches else [None]

    def get_component(self, cls: type | str) -> list:
        """
        Alias for get_instance (instances are instantiated, components are not)
        """
        return self.get_instance(cls)

    def visualize(self, pdf: bool = True) -> None:
        """
        Visualize each compoonent instances either by displaying plots or saving them to a PDF file.

        Parameters:

            pdf (bool): If True, save the plots to a PDF file. If False, display the plots interactively. Default is True.

        Returns:

            None
        """

        if not pdf:
            for instance in self.instances:
                for _plot in instance.plot():
                    plt.show()

        else:
            print("Generating PDF output…")
            pdf_filename = f"{self.name} {self.tstart:%Y-%m-%d %H%M%S}.pdf"
            with PdfPages(pdf_filename) as pdf_file:
                for instance in self.instances:
                    for _plot in instance.plot():
                        pdf_file.savefig()
                        plt.close()

            print(f"PDF output saved to '{pdf_filename}'.")

        return

    def plot(self, fig: Figure | None = None):
        raise NotImplementedError("Subclasses must implement this method")


class BaseComponent(ABC, Generic[ModelType]):
    """
    Base class for all laser-measles components.

    Components follow a uniform interface with __call__(model, tick) method
    for execution during simulation loops.
    """

    def __init__(self, model: BaseLaserModel, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose
        self.initialized = False
        if not hasattr(self, "name"):
            self.name = self.__class__.__name__

    def initialize(self, model: BaseLaserModel) -> None:
        """Initialize component based on other existing components. This is run at the beginning of model.run()."""

    def __str__(self) -> str:
        """Return string representation using class docstring."""
        # Use child class docstring if available, otherwise parent class
        doc = self.__class__.__doc__ or BaseComponent.__doc__
        return doc.strip() if doc else f"{self.__class__.__name__} component"

    def plot(self, fig: Figure | None = None):
        """
        Placeholder for plotting method.
        """
        yield None


class BasePhase(BaseComponent):
    """
    Base class for all laser-measles phases.

    Phases are components that are called every tick and include a __call__ method.
    """

    @abstractmethod
    def __call__(self, model, tick: int) -> None:
        """Execute component logic for a given simulation tick."""


class BaseScenario:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def _validate(self, df: pl.DataFrame):
        # Validate required columns exist - derive from schema
        raise NotImplementedError("Subclasses must implement this method")

    def __getattr__(self, attr):
        # Forward attribute access to the underlying DataFrame
        return getattr(self._df, attr)

    def __getitem__(self, key):
        return self._df[key]

    def __repr__(self):
        return repr(self._df)

    def __len__(self):
        return len(self._df)

    def unwrap(self) -> pl.DataFrame:
        return self._df

    def find_row_number(self, column: str, target_value: str) -> int:
        """
        Find the row number (0-based index) of a target string in a DataFrame column.

        Args:
            column: Column name to search in
            target_value: String value to find

        Returns:
            Row number (0-based index) of the target string

        Raises:
            ValueError: If the target string is not found
        """
        # Use arg_max on a boolean mask for maximum efficiency
        mask = self._df[column] == target_value

        # Check if value exists
        if not mask.any():
            raise ValueError(f"String '{target_value}' not found in column '{column}'")

        # arg_max returns the index of the first True value
        return mask.arg_max()
