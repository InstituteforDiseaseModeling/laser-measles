from pathlib import Path
from collections import defaultdict
from typing import Protocol

import polars as pl
from pydantic import BaseModel
from pydantic import field_validator
from shapefile import Reader


class DemographicsGeneratorProtocol(Protocol):
    def generate_population(self) -> pl.DataFrame: ...

    def generate_birth_rates(self) -> pl.DataFrame: ...

    def generate_mortality_rates(self) -> pl.DataFrame: ...


class ShapefileProtocol(Protocol):
    def add_dotname(self) -> None: ...

    def get_dataframe(self) -> pl.DataFrame: ...


class BaseShapefile(BaseModel):
    shapefile: Path

    @classmethod
    @field_validator("shapefile", mode="before")
    def convert_to_path(cls, v):
        p = Path(v) if not isinstance(v, Path) else v
        if not p.exists():
            raise FileNotFoundError(f"Shapefile {p} does not exist")
        return p

    def add_dotname(self) -> None: ...

    def get_dataframe(self) -> pl.DataFrame:
        """
        Get a Polars DataFrame containing the shapefile data and fields.

        Returns:
            A Polars DataFrame.
        """

        with Reader(self.shapefile) as sf:
            # Get all records and shapes
            records = []
            shapes = []
            for shaperec in sf.iterShapeRecords():
                records.append(shaperec.record)
                shapes.append(shaperec.shape)

            record_dict = defaultdict(list)
            for record in records:
                for key, value in record.as_dict().items():
                    record_dict[key].append(value)

            # Convert to DataFrame
            df = pl.DataFrame(record_dict)

            # Add shape column
            df = df.with_columns(pl.Series(name="shape", values=shapes))

            return df
