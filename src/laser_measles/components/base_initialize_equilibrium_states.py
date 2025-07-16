"""
Component for initializing the population in each of the model states by rough equilibrium of R0.
"""

from pydantic import BaseModel
from pydantic import Field

from laser_measles.base import BaseLaserModel
from laser_measles.base import BasePhase


class BaseInitializeEquilibriumStatesParams(BaseModel):
    """
    Parameters for the InitializeEquilibriumStatesProcess.
    """

    R0: float = Field(default=8.0, description="Basic reproduction number setting the initialization", ge=0.0)


class BaseInitializeEquilibriumStatesProcess(BasePhase):
    """
    Initialize S, R states of the population in each of the model states by rough equilibrium of R0.
    """

    def __init__(self, model: BaseLaserModel, verbose: bool = False, params: BaseInitializeEquilibriumStatesParams | None = None):
        super().__init__(model, verbose)
        self.params = params or BaseInitializeEquilibriumStatesParams()

    def _initialize(self, model: BaseLaserModel):
        """
        Initialize the population in each of the model states by rough equilibrium of R0.
        THis is run after model.run()
        """
        states = model.patches.states
        population = states.S + states.R
        states.S = population / self.params.R0
        states.R = population * (1 - 1 / self.params.R0)
