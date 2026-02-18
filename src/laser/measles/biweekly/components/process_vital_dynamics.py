from laser_measles.components import BaseVitalDynamicsParams
from laser_measles.components import BaseVitalDynamicsProcess


class VitalDynamicsParams(BaseVitalDynamicsParams):
    """
    Parameters for the vital dynamics process.
    """


class VitalDynamicsProcess(BaseVitalDynamicsProcess):
    """
    Phase for simulating the vital dynamics in the model with MCV1.
    """

    def calculate_capacity(self, model) -> int:
        raise RuntimeError("No capacity for this model")
