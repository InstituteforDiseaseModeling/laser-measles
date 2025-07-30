import numpy as np
import polars as pl
from laser_core.migration import stouffer
from pydantic import BaseModel
from pydantic import Field

from laser_measles.mixing.base import BaseMixing


class StoufferParams(BaseModel):
    """
    Parameters for the stouffer migration model.

    Args:
        include_home (bool): Whether to include home in the migration matrix
        k (float): Scale parameter (avg trip probability)

    """

    include_home: bool = Field(default=True, description="Whether to include home in the migration matrix")
    k: float = Field(default=0.01, description="Scale parameter (avg trip probability)", ge=0, le=1)
    a: float = Field(default=1.0, description="Population source exponent")
    b: float = Field(default=1.0, description="Population target exponent")


class StoufferMixing(BaseMixing):
    """
    Stouffer migration model.
    """

    def __init__(self, scenario: pl.DataFrame | None = None, params: StoufferParams | None = None):
        if params is None:
            params = StoufferParams()
        super().__init__(scenario, params)

    def get_migration_matrix(self) -> np.ndarray:
        if len(self.scenario) == 1:
            return np.array([[0.0]])
        distances = self.get_distances()
        mat = stouffer(
            self.scenario["pop"].to_numpy(), distances, k=1.0, include_home=self.params.include_home, a=self.params.a, b=self.params.b
        )  # TODO: find a better k?
        # normalize w/ k
        nrm = self.params.k / (np.sum(mat * self.scenario["pop"].to_numpy()[:, np.newaxis], axis=1) / self.scenario["pop"].to_numpy())
        mat *= nrm
        return mat
