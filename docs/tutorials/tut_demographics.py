# %% [markdown]
# # Initialization using shapefiles and rasters

# %% [markdown]
# The initial conditions of the simulation are dictated by demographics (e.g., population, age distribution, etc.). The laser-measles package provides a number of tools to help you generate demographics for your simulation. In this tutorial, we'll download and process a shapefile of Ethiopia at administrative level 1 boundaries to estimate intitial populations per patch. We will also show how we can sub-divide each boundary shape into roughly equal-area patches.

# %% [markdown]
# ## Setup and plot the shapefile

# %% [markdown]
# laser-measles provides some functionality for downloading and plotting GADM shapefiles. Below we will download the data, print it as a dataframe, and then plot it. Note that we have constructed a `DOTNAME` attribute has the format `COUNTRY:REGION`. The data is located in the local directory.

# %%
from pathlib import Path
from IPython.display import display
from laser_measles.demographics import GADMShapefile, get_shapefile_dataframe, plot_shapefile_dataframe

shapefile = Path("ETH/gadm41_ETH_1.shp")

if not shapefile.exists():
    shp = GADMShapefile.download("ETH", admin_level=1 )
    print("Shapefile is now at", shp.shapefile)
else:
    print("Shapefile already exists")
    shp = GADMShapefile(shapefile=shapefile, admin_level=1)

df = get_shapefile_dataframe(shp.shapefile)
print(df.head(n=2))

plot_shapefile_dataframe(df, plot_kwargs={'facecolor': 'xkcd:sky blue'});

# %% [markdown]
# ## Population calculation

# %% [markdown]
# For the simulation we will want to know the initial number of people in each region. First we'll download our population file (~5.6MB) from worldpop using standard libraries:

# %%
import requests

url = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2010/ETH/eth_ppp_2010_1km_Aggregated_UNadj.tif"
output_path = Path("ETH/eth_ppp_2010_1km_Aggregated_UNadj.tif")

if not output_path.exists():
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

# %% [markdown]
# We use the `RasterPatchGenerator` to sum the population in each of the shapes. This is saved into a dataframe that we can use to initialize a simulation.

# %%
from laser_measles.demographics import RasterPatchParams, RasterPatchGenerator
import sciris as sc
# Setup demographics generator
config = RasterPatchParams(
    id="ETH_ADM1",
    region="ETH",
    shapefile=shp.shapefile,
    population_raster=output_path,
)
generator = RasterPatchGenerator(config)
with sc.Timer() as t:
    # Generate the demographics (in this case the population)
    generator.generate_demographics()
    print(f"Total population: {generator.population['pop'].sum()/1e6:.2f} million") # Should be ~90.5M
generator.population.head(n=2)

# %% [markdown]
# laser-measles demographics uses caching to save results. Now we will run the calculation again with a new instance of the `RasterPatchGenerator`.

# %%
new_generator = RasterPatchGenerator(config)
with sc.Timer() as t:
    # # Generate the demographics (in this case the population)
    new_generator.generate_demographics()
    print(f"Total population: {new_generator.population['pop'].sum()/1e6:.2f} million") # Should be ~90.5M

# %% [markdown]
# You can access the cache directory using the associated module

# %%
from laser_measles.demographics import cache
print(f"Cache directory: {cache.get_cache_dir()}")

# %% [markdown]
# ## Sub-divide the regions

# %% [markdown]
# Now we will generate roughtly equal area patches of 700 km using the original `shp` shapefile. Now each shape has a unique identifier with the form `COUNTRY:REGION:ID`. We will also time how long this takes.

# %%
patch_size = 700 # km
new_shapefile = Path(f"ETH/gadm41_ETH_1_{patch_size}km.shp")

new_shp = GADMShapefile(shapefile=shp.shapefile, admin_level=1)
new_shp.shape_subdivide(patch_size_km=patch_size)
print("Shapefile is now at", new_shp.shapefile)

new_df = get_shapefile_dataframe(new_shp.shapefile)
display(new_df.head(n=2))

import matplotlib.pyplot as plt
plt.figure()
ax = plt.gca()
plot_shapefile_dataframe(new_df, plot_kwargs={'facecolor': 'xkcd:sky blue', 'edgecolor':'gray'}, ax=ax)
plot_shapefile_dataframe(df, plot_kwargs={'fill': False}, ax=ax);


