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

import numpy as np

from laser_measles.base import BaseLaserModel
from laser_measles.biweekly.base import BaseBiweeklyScenario
from laser_measles.biweekly.base import PatchLaserFrame
from laser_measles.biweekly.params import BiweeklyParams
from laser_measles.utils import StateArray
from laser_measles.utils import cast_type


class BiweeklyModel(BaseLaserModel[BaseBiweeklyScenario, BiweeklyParams]):
    """
    A class to represent the biweekly model.

    Args:

        scenario (BaseScenario): A scenario containing the scenario data, including population, latitude, and longitude.
        params (BiweeklyParams): A set of parameters for the model.
        name (str, optional): The name of the model. Defaults to "biweekly".

    Notes:

        This class initializes the model with the given scenario and parameters. The scenario must include the following columns:

            - `id` (string): The name of the patch or location.
            - `pop` (integer): The population count for the patch.
            - `lat` (float degrees): The latitude of the patches (e.g., from geographic or population centroid).
            - `lon` (float degrees): The longitude of the patches (e.g., from geographic or population centroid).
            - `mcv1` (float): The MCV1 coverage for the patches.
    """

    patches: PatchLaserFrame

    def __init__(self, scenario: BaseBiweeklyScenario, params: BiweeklyParams, name: str = "biweekly") -> None:
        """
        Initialize the disease model with the given scenario and parameters.

        Args:

            scenario (BaseScenario): A scenario containing the scenario data, including population, latitude, and longitude.
            params (BiweeklyParams): A set of parameters for the model, including seed, nticks, k, a, b, c, max_frac, cbr, verbose, and pyramid_file.
            name (str, optional): The name of the model. Defaults to "biweekly".

        Returns:

            None
        """
        super().__init__(scenario, params, name)

        # Add patches to the model
        self.patches = PatchLaserFrame(capacity=len(scenario))

        # Create the state vector for each of the patches (3, num_patches)
        self.patches.add_vector_property("states", len(self.params.states))  # S, I, R

        # Wrap the states array with StateArray for attribute access
        self.patches.states = StateArray(self.patches.states, state_names=self.params.states)

        # Start with totally susceptible population
        self.patches.states.S[:] = scenario["pop"]

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

    def infect(self, indices: int | np.ndarray, num_infected: int | np.ndarray) -> None:
        """
        Infects the given nodes with the given number of infected individuals.

        Args:
            indices (int | np.ndarray): The indices of the nodes to infect.
            num_infected (int | np.ndarray): The number of infected individuals to infect.
        """

        self.patches.states[1, indices] += cast_type(num_infected, self.patches.states.dtype)
        self.patches.states[0, indices] -= cast_type(num_infected, self.patches.states.dtype)
        return


# Create an alias for BiweeklyModel as Model
Model = BiweeklyModel
