import importlib

import polars as pl
import pytest

from laser_measles import MEASLES_MODULES


def test_api_import():
    pass


@pytest.mark.parametrize("measles_module", MEASLES_MODULES)
def test_model_import(measles_module):
    scenario = pl.DataFrame({"pop": [1000], "lat": [0.0], "lon": [0.0], "id": ["1"], "mcv1": [0.5]})
    MeaslesModule = importlib.import_module(measles_module)
    Model = MeaslesModule.Model
    Params = MeaslesModule.Params
    model = Model(scenario, Params(), "test_model")
    assert model is not None
