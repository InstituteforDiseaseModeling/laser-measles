"""
Process for setting a static population (no vital dynamics).
"""

import numpy as np
import polars as pl
from pydantic import BaseModel

from laser_measles.abm.model import ABMModel
from laser_measles.base import BaseComponent
from laser_measles.base import BaseLaserModel
from laser_measles.utils import cast_type


class NoBirthsParams(BaseModel):
    """Parameters for the no births process."""


class NoBirthsProcess(BaseComponent[ABMModel]):
    """
    Component for setting the population of the patches to not have births.
    """

    def __init__(self, model: BaseLaserModel, params: NoBirthsParams | None = None, verbose: bool = False) -> None:
        super().__init__(model, verbose)

        if params is None:
            params = NoBirthsParams()
        self.params = params

        return

    def initialize(self, model: ABMModel) -> None:
        """
        Initialize the no births process by setting up the population.
        
        Args:
            model: The ABM model instance to initialize
        """
        # initialize the people laserframe with correct capacity
        model.initialize_people_capacity(self.calculate_capacity(model))
        # people laserframe
        people = model.people
        # scenario dataframe
        scenario = model.scenario
        # initialize the patch ids according to the scenario population
        people.patch_id[:] = np.array(scenario.with_row_index().select(
            pl.col("index").repeat_by(pl.col("pop"))
        ).explode("index")["index"].to_numpy(), dtype=people.patch_id.dtype)
        return

    def calculate_capacity(self, model: ABMModel) -> int:
        """
        Calculate the capacity of the people laserframe.
        
        Args:
            model: The ABM model instance
            
        Returns:
            The total population capacity needed across all patches
        """
        return int(model.patches.states.sum())
