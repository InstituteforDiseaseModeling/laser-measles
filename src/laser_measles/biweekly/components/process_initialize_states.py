"""
The component initializes the population in each of the model states by
rough equilibrium of R0.

"""

from pydantic import BaseModel
from pydantic import Field

from laser_measles.base import BaseComponent
from laser_measles.base import BaseLaserModel


class InitializeEquilibriumStatesParams(BaseModel):
    """
    Parameters for the InitializeEquilibriumStatesProcess.
    """

    R0: float = Field(8.0, description="Basic reproduction number", ge=0.0)


class InitializeEquilibriumStatesProcess(BaseComponent):
    """
    Initialize S, R states of the population in each of the model states by rough equilibrium of R0.
    """

    def __init__(self, model: BaseLaserModel, verbose: bool = False, params: InitializeEquilibriumStatesParams | None = None):
        super().__init__(model, verbose)
        if params is None:
            params = InitializeEquilibriumStatesParams()
        self.params = params

    def initialize(self, model: BaseLaserModel):
        """
        Initialize the population in each of the model states by rough equilibrium of R0.
        
        This is run before model.run() to set initial conditions.
        
        Args:
            model: The model instance to initialize
        """
        states = model.patches.states
        population = states.S + states.R
        states.S = population / self.params.R0
        states.R = population * (1 - 1 / self.params.R0)
