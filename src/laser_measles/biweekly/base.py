"""
Basic classes
"""

import polars as pl
from matplotlib.figure import Figure
from pydantic import BaseModel
from pydantic import ValidationError


class BaseComponent:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose
        self.initialized = True

    def __call__(self, model, tick: int) -> None: ...

    def plot(self, fig: Figure = None):
        """
        Placeholder for plotting method.
        """
        yield None


class BaseScenarioSchema(BaseModel):
    """
    Schema for the scenario data.
    """
    pop: list[int] # population
    lat: list[float] # latitude
    lon: list[float] # longitude
    ids: list[str | int] # ids of the nodes
    mcv1: list[float] # MCV1 coverages


class BaseScenario:
    def __init__(self, df: pl.DataFrame):
        self._df = df
        self._validate(df)

    def _validate(self, df: pl.DataFrame):
        try:
            # Convert to dict of columns for validation
            data_dict = df.to_dict(as_series=False)
            BaseScenarioSchema(**data_dict)
        except ValidationError as e:
            raise ValueError(f"DataFrame validation error:\n{e}") from e

    def __getattr__(self, attr):
        # Forward attribute access to the underlying DataFrame
        return getattr(self._df, attr)

    def __getitem__(self, key):
        return self._df[key]

    def __repr__(self):
        return repr(self._df)

    def __len__(self):
        return len(self._df)

    def unwrap(self) -> pl.DataFrame:
        return self._df
