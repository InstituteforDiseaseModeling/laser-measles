from abc import ABC
from abc import abstractmethod

import numpy as np
import polars as pl


class BaseMixing(ABC):
    """
    Base class for migration models.
    """

    def __init__(self, scenario, params):
        self._scenario = scenario
        self.params = params
        self._migration_matrix = None
        self._mixing_matrix = None

    @property
    def scenario(self) -> pl.DataFrame:
        return self._scenario

    @scenario.setter
    def scenario(self, scenario: pl.DataFrame) -> None:
        self._scenario = scenario

    @property
    def migration_matrix(self) -> np.ndarray:
        """
        Migration matrix computed from get_migration_matrix().

        Returns:
            np.ndarray: The migration matrix with lazy computation and caching.
        """
        if self._migration_matrix is None:
            self._migration_matrix = self.get_migration_matrix()
        return self._migration_matrix

    @property
    def mixing_matrix(self) -> np.ndarray:
        """
        Mixing matrix computed from get_mixing_matrix().

        Returns:
            np.ndarray: The mixing matrix with lazy computation and caching.
        """
        if self._mixing_matrix is None:
            self._mixing_matrix = self.get_mixing_matrix()
        return self._mixing_matrix

    @abstractmethod
    def get_migration_matrix(self) -> np.ndarray:
        """
        Initialize a migration/diffusion matrix for population mixing. The diffusion
        matrix is a square matrix where each row represents the outbound migration
        from a given patch to all other patches e.g., [i,j] = [from_i, to_j].

        Convention is:
        - Trips into node j: N_i @ M[i,j]
        - Trips out of node i: np.sum(M[i,j] * N_i[:,np.newaxis], axis=1)

        Returns:
            np.ndarray: The diffusion matrix: (N, N)
        """
        ...

    @abstractmethod
    def get_mixing_matrix(self) -> np.ndarray:
        """
        Initialize a mixing matrix for population mixing.

        The mixing matrix is a square matrix where each row represents the
        mixing of a given patch to all other patches e.g., [i,j] = [from_i, to_j].
        It also includes internal mixing within a patch.
        """
        ...

    def trips_into(self) -> np.ndarray:
        """Returns the number of trips into each patch per tick."""
        return self.scenario["pop"].to_numpy() @ self.migration_matrix

    def trips_out_of(self) -> np.ndarray:
        """Returns the number of trips out of each patch per tick."""
        return np.sum(self.migration_matrix * self.scenario["pop"].to_numpy()[:, np.newaxis], axis=1)
