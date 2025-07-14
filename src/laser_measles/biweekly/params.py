import json
from collections import OrderedDict

from laser_measles.base import BaseModelParams

TIME_STEP_DAYS = 14
STATES = ["S", "I", "R"]  # Compartments/states for discrete-time model


class BiweeklyParams(BaseModelParams):
    """
    Parameters for the biweekly model.
    """

    @property
    def time_step_days(self) -> int:
        return TIME_STEP_DAYS

    @property
    def states(self) -> list[str]:
        return STATES

    def __str__(self) -> str:
        return json.dumps(OrderedDict(sorted(self.model_dump().items())), indent=2)


Params = BiweeklyParams
