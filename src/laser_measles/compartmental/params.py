import json
from collections import OrderedDict

from laser_measles.base import BaseModelParams

TIME_STEP_DAYS = 1
STATES = ["S", "E", "I", "R"]  # Compartments/states for SEIR model


class CompartmentalParams(BaseModelParams):
    """
    Parameters for the compartmental SEIR model with daily timesteps.
    """

    @property
    def time_step_days(self) -> int:
        return TIME_STEP_DAYS

    @property
    def states(self) -> list[str]:
        return STATES

    def __str__(self) -> str:
        return json.dumps(OrderedDict(sorted(self.model_dump().items())), indent=2)


Params = CompartmentalParams
