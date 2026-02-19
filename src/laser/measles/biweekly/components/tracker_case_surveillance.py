from laser.measles.components import BaseCaseSurveillanceParams
from laser.measles.components import BaseCaseSurveillanceTracker


class CaseSurveillanceParams(BaseCaseSurveillanceParams): ...


class CaseSurveillanceTracker(BaseCaseSurveillanceTracker):
    """Tracks detected cases in the model."""
