import numpy as np
import polars as pl
from laser_core.migration import competing_destinations
from pydantic import BaseModel
from pydantic import Field

from laser_measles.mixing.base import BaseMixing


class CompetingDestinationsParams(BaseModel):
    """
    Parameters for the competing destinations mixing model.
    """

    a: float = Field(default=1.0, description="Population source scale parameter", ge=1.0)
    b: float = Field(default=1.0, description="Population target scale parameter")
    c: float = Field(default=1.5, description="Distance exponent")
    k: float = Field(default=0.01, description="Scale parameter (avg trip probability)", ge=0, le=1)
    delta: float = Field(default=0.0, description="Destination selection parameter")


class CompetingDestinationsMixing(BaseMixing):
    """
    Competing destinations mixing model that accounts for the effects of nearby destinations.

    Formula:
        .. math::
            M_{i,j} = k \\frac{p_i^{a-1} p_j^b}{d_{i,j}^c} \\left(\\sum_{k \\ne i,j} \\frac{p_k^b}{d_{ik}^c}\\right)^\\delta

    Where:
        - M_{i,j}: migration flow from origin i to destination j
        - k: calibration constant
        - p_i, p_j, p_k: population at origins/destinations
        - d_{i,j}, d_{ik}: distances between l  ocations
        - a, b, c, δ: model parameters
    """

    def __init__(self, scenario: pl.DataFrame | None = None, params: CompetingDestinationsParams | None = None):
        if params is None:
            params = CompetingDestinationsParams()
        super().__init__(scenario, params)

    def get_migration_matrix(self) -> np.ndarray:
        if len(self.scenario) == 1:
            return np.array([[0.0]])
        distances = self.get_distances()
        mat = competing_destinations(
            self.scenario["pop"].to_numpy(),
            distances,
            k=1.0,
            a=self.params.a - 1,
            b=self.params.b,
            c=self.params.c,
            delta=self.params.delta,
        )  # TODO: find a better k?
        # normalize w/ k
        nrm = self.params.k / (np.sum(mat * self.scenario["pop"].to_numpy()[:, np.newaxis], axis=1) / self.scenario["pop"].to_numpy())
        mat *= nrm
        return mat
