"""
Basic classes for biweekly model.
"""

import numpy as np
import patito as pt
import polars as pl

from laser_measles.base import BasePatchLaserFrame
from laser_measles.base import BaseScenario


class PatchLaserFrame(BasePatchLaserFrame): ...


class BaseScenarioSchema(pt.Model):
    """
    Schema for the scenario data.
    """

    pop: int  # population
    lat: float  # latitude
    lon: float  # longitude
    id: str  # ids of the nodes
    mcv1: float  # MCV1 coverages (as percentages, will be divided by 100)


class BaseBiweeklyScenario(BaseScenario):
    def __init__(self, df: pl.DataFrame):
        super().__init__(df)
        BaseScenarioSchema.validate(df, allow_superfluous_columns=True)

    def _validate(self, df: pl.DataFrame):
        # Validate required columns exist
        required_columns = ["pop", "lat", "lon", "id", "mcv1"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate data types using Polars' native operations
        try:
            # Validate pop is integer
            if not df["pop"].dtype == pl.Int64:
                raise ValueError("Column 'pop' must be integer type")

            # Validate lat and lon are float
            if not df["lat"].dtype == pl.Float64:
                raise ValueError("Column 'lat' must be float type")
            if not df["lon"].dtype == pl.Float64:
                raise ValueError("Column 'lon' must be float type")

            # Validate mcv1 is float
            if not df["mcv1"].dtype == pl.Float64:
                raise ValueError("Column 'mcv1' must be float type")

            # Validate mcv1 is between 0 and 1 (as percentages)
            if not df["mcv1"].is_between(0, 1).all():
                raise ValueError("Column 'mcv1' must be between 0 and 1")

            # Validate ids are either string or integer
            if not (df["id"].dtype == pl.String or df["id"].dtype == pl.Int64):
                raise ValueError("Column 'id' must be either string or integer type")

            # Validate no null values
            null_counts = df.null_count()
            if np.any(null_counts):
                raise ValueError(f"DataFrame contains null values:\n{null_counts}")

        except Exception as e:
            raise ValueError(f"DataFrame validation error:\n{e}") from e


BaseScenario = BaseBiweeklyScenario
