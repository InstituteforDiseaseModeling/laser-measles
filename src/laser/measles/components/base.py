from abc import abstractmethod

from laser.measles.base import BaseLaserModel
from laser.measles.base import BasePhase


class BaseVitalDynamicsProcess(BasePhase):
    @abstractmethod
    def initialize(self, model: BaseLaserModel) -> None: ...

    @abstractmethod
    def calculate_capacity(self, model: BaseLaserModel) -> int: ...
