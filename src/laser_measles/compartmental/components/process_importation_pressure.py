"""
Component for simulating the importation pressure in the compartmental model.
"""

import numpy as np
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from laser_measles.base import BasePhase
from laser_measles.utils import cast_type


class ImportationPressureParams(BaseModel):
    """Parameters specific to the importation pressure component."""

    crude_importation_rate: float = Field(default=1.0, description="Yearly crude importation rate per 1k population", ge=0.0)
    importation_start: int = Field(default=0, description="Start time for importation (in days)", ge=0)
    importation_end: int = Field(default=-1, description="End time for importation (in days)", ge=-1)

    @field_validator("importation_end")
    @classmethod
    def validate_importation_end(cls, v, info):
        """Validate that importation_end is greater than importation_start when not -1."""
        if v != -1:
            start = info.data.get("importation_start", 0)
            if v <= start:
                raise ValueError("importation_end must be greater than importation_start")
        return v


class ImportationPressureProcess(BasePhase):
    """
    Component for simulating the importation pressure in the model.

    This component handles the simulation of disease importation into the population.
    It processes:
    - Importation of cases based on crude importation rate
    - Time-windowed importation (start/end times)
    - Population updates: Moves individuals from susceptible to infected state

    Parameters
    ----------
    model : object
        The simulation model containing nodes, states, and parameters
    verbose : bool, default=False
        Whether to print verbose output during simulation
    params : Optional[ImportationPressureParams], default=None
        Component-specific parameters. If None, will use default parameters

    Notes
    -----
    - Importation rates are calculated per year
    - Importation is limited to the susceptible population
    - All state counts are ensured to be non-negative
    """

    def __init__(self, model, verbose: bool = False, params: ImportationPressureParams | None = None) -> None:
        super().__init__(model, verbose)
        self.params = params or ImportationPressureParams(crude_importation_rate=1.0, importation_start=0, importation_end=-1)

    def __call__(self, model, tick: int) -> None:
        if tick < (self.params.importation_start // model.params.time_step_days) or (
            self.params.importation_end != -1 and tick > (self.params.importation_end // model.params.time_step_days)
        ):
            return

        # state counts
        states = model.patches.states

        # population
        population = states.sum(axis=0, dtype=np.int64)  # promote to int64, otherwise binomial draw will fail

        # Sample actual number of imported cases
        imported_cases = model.prng.binomial(population, (self.params.crude_importation_rate / 365.0 / 1000.0))
        imported_cases = cast_type(imported_cases, states.dtype)
        np.minimum(imported_cases, states.S, out=imported_cases)

        # update states
        states.S -= imported_cases
        states.E += imported_cases  # Move to exposed state
