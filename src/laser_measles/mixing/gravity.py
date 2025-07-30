import numpy as np
import polars as pl
from laser_core.migration import gravity
from pydantic import BaseModel
from pydantic import Field

from laser_measles.mixing.base import BaseMixing


class GravityParams(BaseModel):
    """
    Parameters for the gravity migration model.

    network = k * (pops[:, np.newaxis] ** (a-1)) * (pops ** b) * (distances ** (-1 * c))

    Defaults scale linearly with target node.

    Args:
        a (float): Population source scale parameter
        b (float): Population target scale parameter
        c (float): Distance exponent
        k (float): Scale parameter
    """

    a: float = Field(default=1.0, description="Population source scale parameter", ge=1.0)
    b: float = Field(default=1.0, description="Population target scale parameter")
    c: float = Field(default=1.5, description="Distance exponent")
    k: float = Field(default=0.01, description="Scale parameter (avg trip probability)", ge=0, le=1)


class GravityMixing(BaseMixing):
    """
    Gravity migration model.

    network = (pops_from[:, np.newaxis] ** (a-1)) * (pops_to ** b) * (distances ** (-1 * c))

    """

    def __init__(self, scenario: pl.DataFrame | None = None, params: GravityParams | None = None):
        if params is None:
            params = GravityParams()
        super().__init__(scenario, params)

    def get_migration_matrix(self) -> np.ndarray:
        if len(self.scenario) == 1:
            return np.array([[0.0]])
        distances = self.get_distances()
        mat = gravity(
            self.scenario["pop"].to_numpy(), distances, k=1.0, a=self.params.a - 1, b=self.params.b, c=self.params.c
        )  # TODO: find a better k?
        # normalize w/ k
        nrm = self.params.k / (np.sum(mat * self.scenario["pop"].to_numpy()[:, np.newaxis], axis=1) / self.scenario["pop"].to_numpy())
        mat *= nrm
        return mat
