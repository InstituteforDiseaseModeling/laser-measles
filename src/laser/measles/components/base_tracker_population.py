import numpy as np
from pydantic import BaseModel

from laser_measles.base import BaseLaserModel
from laser_measles.base import BasePhase


class BasePopulationTrackerParams(BaseModel):
    pass


class BasePopulationTracker(BasePhase):
    """
    Tracks the population size of each patch at each time tick.
    """

    def __init__(self, model: BaseLaserModel, verbose: bool = False, params: BasePopulationTrackerParams | None = None) -> None:
        super().__init__(model, verbose)
        self.params = params or BasePopulationTrackerParams()
        self.population_tracker = np.zeros((model.patches.count, model.params.num_ticks), dtype=model.patches.states.dtype)

    def __call__(self, model, tick: int) -> None:
        self.population_tracker[:, tick] = model.patches.states.sum(axis=0)

    def _initialize(self, model: BaseLaserModel) -> None:
        pass
