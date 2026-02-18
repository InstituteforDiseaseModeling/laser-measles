from abc import abstractmethod

from laser_measles.base import BaseLaserModel
from laser_measles.base import BasePhase


class BaseVitalDynamicsProcess(BasePhase):
    @abstractmethod
    def initialize(self, model: BaseLaserModel) -> None: ...

    @abstractmethod
    def calculate_capacity(self, model: BaseLaserModel) -> int: ...
