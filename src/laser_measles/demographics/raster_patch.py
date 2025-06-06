"""
Raster patch generator for demographic data. 
You can use this to generate initial conditions for a laser-measles scenario.
"""

from pathlib import Path

import alive_progress
import numpy as np
import polars as pl
from PIL import Image
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from rastertoolkit import raster_clip
from rastertoolkit import raster_clip_weighted

from laser_measles.demographics import cache
from laser_measles.demographics import shapefiles
from laser_measles.demographics.gadm import GADMShapefile


class RasterPatchConfig(BaseModel):
    id: str = Field(..., description="Unique identifier for the scenario")
    region: str = Field(..., description="Country identifier (ISO3 code)")
    shapefile_path: str | Path = Field(..., description="Path to the shapefile")
    population_raster_path: str | Path = Field(..., description="Path to the population raster")
    mcv1_raster_path: str | Path | None = Field(None, description="Path to the MCV1 raster")
    mcv2_raster_path: str | Path | None = Field(None, description="Path to the MCV2 raster")

    @field_validator("shapefile_path")
    def shapefile_path_exists(cls, v, info):
        path = Path(v) if isinstance(v, str) else v
        if not path.exists():
            raise ValueError(f"Shapefile path does not exist: {path}")
        return v


class RasterPatchGenerator:
    def __init__(self, config: RasterPatchConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.population = None
        self.mcv1 = None
        self.mcv2 = None
        self._validate_config()

    def generate_demographics(self) -> None:
        self._validate_shapefile()
        self.population = self.generate_population()
        if self.config.mcv1_raster_path is not None:
            self.mcv1 = self.generate_mcv1()

    def _validate_config(self) -> None:
        if not shapefiles.check_field(self.config.shapefile_path, "DOTNAME"):
            raise ValueError(f"Shapefile {self.config.shapefile_path} does not have a DOTNAME field")

    def _validate_shapefile(self):
        """ """
        path = Path(self.config.shapefile_path) if isinstance(self.config.shapefile_path, str) else self.config.shapefile_path
        if not path.exists():
            raise FileNotFoundError(f"Shapefile path does not exist: {path}")
        if not shapefiles.check_field(path, "DOTNAME"):
            raise ValueError(f"Shapefile {path} does not have a DOTNAME field")
        self.shapefile = path

    def get_cache_key(self, key) -> str:
        keys = ["shapefile", "population", "mcv1", "mcv2"]
        if key not in keys:
            raise ValueError(f"Invalid key: {key}\nValid keys are: {keys}")
        return f"{self.config.id}" + ":" + key

    def generate_population(self) -> pl.DataFrame:
        with cache.load_cache() as c:
            if self.get_cache_key("population") not in c:
                # clip the raster to the shapefile
                with alive_progress.alive_bar(title="Clipping population raster to shapefile"):
                    popdict = raster_clip(self.config.population_raster_path, self.shapefile, include_latlon=True)
                    new_dict = {"dotname": [], "lat": [], "lon": [], "pop": []}
                    for k, v in popdict.items():
                        new_dict["dotname"].append(k)
                        new_dict["lat"].append(v["lat"])
                        new_dict["lon"].append(v["lon"])
                        new_dict["pop"].append(v["pop"])
                    c[self.get_cache_key("population")] = new_dict
            new_dict = c[self.get_cache_key("population")]
            return pl.DataFrame(new_dict)

    def generate_mcv1(self) -> pl.DataFrame:
        """MCV1 coverage, population weighted"""

        with cache.load_cache() as c:
            if self.get_cache_key("mcv1") not in c:
                # Value array: Set negative values to zero
                new_values_raster_file = self.config.mcv1_raster_path.with_name(f"{self.config.mcv1_raster_path.stem}_zeros.tif")
                with Image.open(self.config.mcv1_raster_path) as raster:
                    data = np.array(raster)
                    data[data < 0] = 0
                    new_raster = Image.fromarray(data, mode=raster.mode)
                    new_raster.info.update(raster.info)  # Preserve metadata
                    new_raster.save(new_values_raster_file, tiffinfo=raster.tag_v2)

                # Weight array: Set negative values to zero
                new_weight_raster_file = self.config.population_raster_path.with_name(
                    f"{self.config.population_raster_path.stem}_zeros.tif"
                )
                with Image.open(self.config.population_raster_path) as raster:
                    data = np.array(raster)
                    data[data < 0] = np.nan
                    new_raster = Image.fromarray(data, mode=raster.mode)
                    new_raster.info.update(raster.info)  # Preserve metadata
                    new_raster.save(new_weight_raster_file, tiffinfo=raster.tag_v2)

                with alive_progress.alive_bar(title="Clipping MCV1 raster to shapefile"):
                    mcv_dict = raster_clip_weighted(
                        new_weight_raster_file,
                        new_values_raster_file,
                        shape_stem=self.config.shapefile_path,
                        include_latlon=True,
                        weight_summary_func=np.mean,
                    )
                    c[self.get_cache_key("mcv1")] = mcv_dict

                # remove the rasters
                new_weight_raster_file.unlink()
                new_values_raster_file.unlink()

        mcv_dict = c[self.get_cache_key("mcv1")]
        new_dict = {"dotname": [], "lat": [], "lon": [], "mcv1": []}
        for k, v in mcv_dict.items():
            new_dict["dotname"].append(k)
            new_dict["lat"].append(v["lat"])
            new_dict["lon"].append(v["lon"])
            new_dict["mcv1"].append(v["pop"])

        return pl.DataFrame(new_dict)

    def generate_birth_rates(self) -> pl.DataFrame: ...

    def generate_mortality_rates(self) -> pl.DataFrame: ...

    def _add_dotname(self) -> None: ...

if __name__ == "__main__":
    gadm = GADMShapefile("NGA")
    gadm.clear_cache()
    gadm.download()
    gadm.add_dotnames()
    config = RasterPatchConfig(
        region="NGA",
        start_year=2000,
        end_year=2020,
        granularity="patch",
        patch_size_km=25,
        shapefile_path=gadm.get_shapefile_path(2),
        population_raster_path=gadm.shapefile_dir,
    )
    generator = RasterPatchGenerator(config)
    generator.generate_demographics()
    # print(generator.generate_population())
