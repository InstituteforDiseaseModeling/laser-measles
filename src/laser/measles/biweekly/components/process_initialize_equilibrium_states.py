"""
Component for initializing the population in each of the model states by rough equilibrium of R0.
"""

from laser_measles.components import BaseInitializeEquilibriumStatesParams
from laser_measles.components import BaseInitializeEquilibriumStatesProcess


class InitializeEquilibriumStatesParams(BaseInitializeEquilibriumStatesParams):
    """
    Parameters for the InitializeEquilibriumStatesProcess.
    """


class InitializeEquilibriumStatesProcess(BaseInitializeEquilibriumStatesProcess):
    """
    Initialize S, R states of the population in each of the model states by rough equilibrium of R0.
    """
