from laser_measles.components import BaseFadeOutTracker
from laser_measles.components import BaseFadeOutTrackerParams


class FadeOutTracker(BaseFadeOutTracker):
    """A component that tracks the number of nodes experiencing fade-outs over time."""


class FadeOutTrackerParams(BaseFadeOutTrackerParams):
    """Parameters for the FadeOutTracker component."""
