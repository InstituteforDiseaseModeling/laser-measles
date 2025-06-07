import os
from pathlib import Path

import alive_progress
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from laser_measles.demographics import gadm
from laser_measles.demographics import raster_patch
from laser_measles.demographics import shapefiles


def summarize_scenario(df: pl.DataFrame) -> None:
    """Summarize the scenario."""
    # Get the total population
    total_population = df["pop"].sum()
    num_with_pop = (df["pop"] > 0).sum()
    print(f"Total population: {total_population}")
    print(f"Number of patches: {len(df)} ({num_with_pop} with pop > 0)")
    return total_population


# Change directory
os.chdir(Path.resolve(Path(__file__)).parent.as_posix())
this_dir = Path(__file__).parent

# Set rasters
population_raster_path = "gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2010_30_sec_crop.tif"
mcv1_raster_path = "mcv1_cov_mean_raked_2000_2023_11_crop.tif"

# Download shapefile
g_admin2 = gadm.GADMShapefile.download("NGA", 2)
# subdivide
shapefile = g_admin2.shape_subdivide(patch_size_km=50)
g_admin1 = gadm.GADMShapefile(shapefile=shapefile.parent / "gadm41_NGA_1.shp")

# Setup demographics generator
config = raster_patch.RasterPatchConfig(
    id="nigeria_50km",
    region="NGA",
    shapefile_path=shapefile,
    population_raster_path=this_dir / population_raster_path,
    mcv1_raster_path=this_dir / mcv1_raster_path,
)
generator = raster_patch.RasterPatchGenerator(config)
generator.clear_cache()
generator.generate_demographics()

# Grab the data
df_pop = generator.population  # TODO: save everythign in a dataframe
df_mcv1 = generator.mcv1

# Get dataframe with the patches
df = shapefiles.get_dataframe(shapefile).rename({"DOTNAME": "dotname"}).join(df_pop, on="dotname", how="left")
df = df.join(df_mcv1.drop(["lat", "lon"]), on="dotname", how="left")
assert len(df) == len(df_pop)
df_states = g_admin1.get_dataframe().rename({"DOTNAME": "dotname"})

# Add "state" column
df = df.with_columns(pl.col("dotname").str.split_exact(":", 2).struct.field("field_1").alias("state"))
df_states = df_states.with_columns(pl.col("dotname").str.split_exact(":", 1).struct.field("field_1").alias("state"))

with alive_progress.alive_bar() as bar:
    bar.text("Plotting Kano")
    # Plot just Kano
    states = ["kano"]
    filtered_df_kano = df.lazy().filter(pl.col("state").is_in(states)).collect()
    filtered_df_kano = filtered_df_kano.with_columns(pl.lit("kano").alias("scenario_id"))
    fig = shapefiles.plot_dataframe(filtered_df_kano)
    fig.savefig("kano.png", dpi=300, bbox_inches="tight", transparent=True)
    kano_pop = summarize_scenario(filtered_df_kano)

    # Plot Kano plus surrounding states
    bar.text("Plotting Kano region")
    states = [k.lower() for k in ["Kano", "Katsina", "Jigawa", "Kaduna", "Bauchi", "Plateau"]]
    filtered_df_kano_region = df.lazy().filter(pl.col("state").is_in(states)).collect()
    filtered_df_kano_region = filtered_df_kano_region.with_columns(pl.lit("kano_region").alias("scenario_id"))
    filtered_df_states = df_states.lazy().filter(pl.col("state").is_in(states)).collect()
    fig = shapefiles.plot_dataframe(filtered_df_kano_region, plot_kwargs={"linewidth": 0.25})
    shapefiles.plot_dataframe(filtered_df_states, ax=fig.axes[0], plot_kwargs={"linewidth": 1})
    fig.savefig("kano_region.png", dpi=300, bbox_inches="tight", transparent=True)
    kano_region_pop = summarize_scenario(filtered_df_kano_region)

    # Plot Northern nigeria states
    bar.text("Plotting Northern Nigeria")
    states = [
        k.lower()
        for k in [
            "Borno",
            "Bauchi",
            "Gombe",
            "Taraba",
            "Yobe",
            "Adamawa",
            "Jigawa",
            "Kano",
            "Kaduna",
            "Katsina",
            "Kebbi",
            "Sokoto",
            "Zamfara",
            "Niger",
            "Plateau",
            "Nasarawa",
            "Benue",
            "Kogi",
            "Kwara",
            "Federal Capital Territory",
        ]
    ]
    filtered_df_northern = df.lazy().filter(pl.col("state").is_in(states)).collect()
    filtered_df_northern = filtered_df_northern.with_columns(pl.lit("northern_nigeria").alias("scenario_id"))
    filtered_df_states = df_states.lazy().filter(pl.col("state").is_in(states)).collect()
    fig = shapefiles.plot_dataframe(filtered_df_northern, plot_kwargs={"linewidth": 0.1})
    shapefiles.plot_dataframe(filtered_df_states, ax=fig.axes[0], plot_kwargs={"linewidth": 1, "fill": True, "facecolor": "none"})
    fig.savefig("nigeria_region.png", dpi=300, bbox_inches="tight", transparent=True)
    northern_pop = summarize_scenario(filtered_df_northern)

    ###############################

    figsize = (4, 5)

    def format_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_ticks_position("left")

    # Create population comparison bar chart
    plt.figure(figsize=figsize)
    scenarios = ["Kano", "Kano Region", "Northern Nigeria"]
    populations = [kano_pop, kano_region_pop, northern_pop]

    # Create bar plot with seaborn for better aesthetics
    sns.set_style("whitegrid")
    ax = sns.barplot(x=scenarios, y=populations)

    # Customize the plot
    # plt.title('Population Comparison Across Scenarios', pad=20)
    plt.ylabel("Population (millions)", fontsize=12)

    # Format y-axis to show millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x / 1e6:.0f}"))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=12)

    # Remove top and right spines
    format_axes(ax)

    plt.tight_layout()
    plt.savefig("population_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create second figure for non-zero population entries
    plt.figure(figsize=figsize)
    # Calculate number of connections (n choose 2) for each scenario
    connection_counts = [
        (filtered_df_kano["pop"] > 0).sum() * ((filtered_df_kano["pop"] > 0).sum() - 1) // 2,
        (filtered_df_kano_region["pop"] > 0).sum() * ((filtered_df_kano_region["pop"] > 0).sum() - 1) // 2,
        (filtered_df_northern["pop"] > 0).sum() * ((filtered_df_northern["pop"] > 0).sum() - 1) // 2,
    ]

    # Create bar plot with seaborn using lavender color
    sns.set_style("whitegrid")
    ax = sns.barplot(x=scenarios, y=connection_counts, color="#B19CD9")  # Darker lavender color

    # Customize the plot
    plt.ylabel("Number of Connections", fontsize=12)

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=12)

    # Remove top and right spines
    format_axes(ax)

    plt.tight_layout()
    plt.savefig("connection_count_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Combine all scenarios and save to disk
    combined_df = pl.concat([filtered_df_kano, filtered_df_kano_region, filtered_df_northern])
    combined_df.drop("shape").write_parquet("scenarios.parquet")
