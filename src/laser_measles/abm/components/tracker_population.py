import numpy as np

from laser_measles.base import BasePhase

class PopulationTracker(BasePhase):
    """
    Tracks the population size of each patch at each time tick.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        self.population_tracker = np.zeros((model.patches.count, model.params.num_ticks), dtype=model.patches.states.dtype)

    def __call__(self, model, tick: int) -> None:
        self.population_tracker[:,tick] = model.patches.states.sum(axis=0)