"""
Admin level shapefiles
"""

from pathlib import Path
from typing import ClassVar

import alive_progress
import polars as pl
from rastertoolkit import shape_subdivide
from shapefile import Reader

from laser_measles.demographics import shapefiles
from laser_measles.demographics.base import BaseShapefile


class AdminShapefile(BaseShapefile):
    dotname_fields: list[str] | None = None  # List of fields to use for dotname. e.g., []

    def get_shapefile_parent(self) -> Path:
        """Get the parent directory of the shapefile."""
        return self.shapefile.parent

    def add_dotname(self) -> None:
        """Add a DOTNAME to the shapefile"""
        shapefiles.add_dotname(self.shapefile, dot_name_fields=self.dotname_fields, inplace=True)

    def shape_subdivide(
        self,
        patch_size_km: int,
    ) -> None:
        """Subdivide the GADM shapefile for a given admin level into patches of a given size."""

        out_file = self.shapefile.parent / f"{self.shapefile.stem}_{patch_size_km}km.shp"
        if out_file.exists():
            return out_file
        else:
            # Add dotname if it doesn't exist
            if not shapefiles.check_field(self.shapefile, "DOTNAME"):
                self.add_dotname()
            with alive_progress.alive_bar(
                title=f"Subdividing shapefile {self.shapefile.stem}",
            ) as _:
                shape_subdivide(
                    shape_stem=self.shapefile,
                    out_dir=self.get_shapefile_parent(),
                    out_suffix=f"{patch_size_km}km",
                    box_target_area_km2=patch_size_km,
                )
                return out_file
