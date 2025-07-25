from laser_measles.components import BasePopulationTracker
from laser_measles.components import BasePopulationTrackerParams


class PopulationTracker(BasePopulationTracker):
    """Tracks the population size of each patch."""


class PopulationTrackerParams(BasePopulationTrackerParams): ...
