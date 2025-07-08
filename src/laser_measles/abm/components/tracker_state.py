from laser_measles.components import BaseStateTracker
from laser_measles.components import BaseStateTrackerParams


class StateTrackerParams(BaseStateTrackerParams):
    """
    Parameters for ABM state tracking component.

    Inherits all parameters from BaseStateTrackerParams with
    ABM-specific defaults and validation.
    """


class StateTracker(BaseStateTracker):
    """
    ABM state tracking component.

    Tracks disease state populations over time in agent-based models.
    Records detailed temporal dynamics of S, E, I, R compartments
    at the patch level.
    """
