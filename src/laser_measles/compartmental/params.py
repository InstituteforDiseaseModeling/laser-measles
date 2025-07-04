import json
from collections import OrderedDict

from pydantic import BaseModel
from pydantic import Field

TIME_STEP_DAYS = 1
STATES = ["S", "E", "I", "R"]  # Compartments/states for SEIR model

class CompartmentalParams(BaseModel):
    """
    Parameters for the compartmental SEIR model with daily timesteps.
    """

    num_ticks: int = Field(..., description="Number of time steps (days)")
    seed: int = Field(default=20250314, description="Random seed")
    start_time: str = Field(default="2005-01", description="Initial start time of simulation in YYYY-MM format")
    verbose: bool = Field(default=False, description="Whether to print verbose output")

    @property
    def time_step_days(self) -> int:
        return TIME_STEP_DAYS

    @property
    def states(self) -> list[str]:
        return STATES

    def __str__(self) -> str:
        return json.dumps(OrderedDict(sorted(self.model_dump().items())), indent=2)

Params = CompartmentalParams