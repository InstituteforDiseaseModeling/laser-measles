from laser.measles.components import BaseFadeOutTracker
from laser.measles.components import BaseFadeOutTrackerParams


class FadeOutTracker(BaseFadeOutTracker):
    """A component that tracks the number of nodes experiencing fade-outs over time."""

    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)


class FadeOutTrackerParams(BaseFadeOutTrackerParams):
    """Parameters for the FadeOutTracker component."""
