from laser.measles.components import BasePopulationTracker
from laser.measles.components import BasePopulationTrackerParams


class PopulationTracker(BasePopulationTracker):
    """Tracks the population size of each patch at each time tick."""


class PopulationTrackerParams(BasePopulationTrackerParams): ...
