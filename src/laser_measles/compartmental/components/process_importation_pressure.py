import numpy as np
from pydantic import BaseModel
from pydantic import Field

from laser_measles.base import BasePhase
from laser_measles.base import BaseLaserModel
from laser_measles.utils import cast_type


class ImportationPressureParams(BaseModel):
    """Parameters specific to the importation pressure component."""

    crude_importation_rate: float = Field(1.0, description="Yearly crude importation rate per 1k population", ge=0.0)
    importation_start: int = Field(0, description="Start time for importation (in ticks)", ge=0)
    importation_end: int = Field(-1, description="End time for importation (in ticks)", ge=-1)


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
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate component parameters."""
        if self.params.importation_end != -1 and self.params.importation_end <= self.params.importation_start:
            raise ValueError("importation_end must be greater than importation_start")

        if self.params.crude_importation_rate < 0:
            raise ValueError("crude_importation_rate must be non-negative")

    def __call__(self, model, tick: int) -> None:
        if tick < self.params.importation_start or (self.params.importation_end != -1 and tick > self.params.importation_end):
            return

        # state counts
        states = model.patches.states

        # population
        population = states.sum(axis=0, dtype=np.int64) # promote to int64, otherwise binomial draw will fail

        # Sample actual number of imported cases
        imported_cases = model.prng.binomial(population, (self.params.crude_importation_rate / 365.0 / 1000.0))
        imported_cases = cast_type(imported_cases, states.dtype)
        np.minimum(imported_cases, states.S, out=imported_cases)

        # update states
        states.S -= imported_cases
        states.I += imported_cases  # Move to infected state

    def initialize(self, model: BaseLaserModel) -> None:
        pass