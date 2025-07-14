"""Synthetic scenarios for testing and development."""

import numpy as np
import polars as pl


def single_patch_scenario(population: int = 100_000, mcv1_coverage: float = 0.0) -> pl.DataFrame:
    """Generate a synthetic scenario with a single patch.

    Args:
        population (int): Population of the patch.
        mcv1_coverage (float): MCV1 coverage of the patch.

    Returns:
        pl.DataFrame: Scenario DataFrame.
    """
    df = pl.DataFrame({"id": ["patch_1"], "pop": [population], "lat": [40.0], "lon": [4.0], "mcv1": [mcv1_coverage]})
    return df


def two_patch_scenario(population: int = 100_000, mcv1_coverage: float = 0.0) -> pl.DataFrame:
    """Generate a synthetic scenario with two patches where one is half the size of the other.

    Args:
        population (int): Population of the largest patch.
        mcv1_coverage (float): MCV1 coverage of the patches.

    Returns:
        pl.DataFrame: Scenario DataFrame.
    """
    df = pl.DataFrame(
        {
            "id": ["patch_1", "patch_2"],
            "pop": [population, population // 2],
            "lat": [40.0, 34.0],
            "lon": [4.0, 10.0],
            "mcv1": [mcv1_coverage, mcv1_coverage],
        }
    )
    return df


def two_cluster_scenario(
    seed: int = 42,
    n_nodes_per_cluster: int = 50,
    cluster_centers: list[tuple[float, float]] | None = None,
    cluster_size_std: float = 0.3,
    mcv1_coverage_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Generate a synthetic scenario with two clusters of nodes.

    Args:
        seed (int): Random seed for reproducibility.
        n_nodes_per_cluster (int): Number of nodes per cluster.
        cluster_centers (list[tuple[float, float]]): List of tuples representing the centers of the clusters.
        cluster_size_std (float): Standard deviation of the Gaussian distribution for cluster size.
        mcv1_coverage_range (tuple[float, float]): Range of MCV1 coverage percentages.

    Returns:
        pl.DataFrame: Scenario DataFrame.
    """

    # Set defaults for mutable arguments
    if cluster_centers is None:
        cluster_centers = [(40.0, 4.0), (34.0, 10.0)]
    if mcv1_coverage_range is None:
        mcv1_coverage_range = (0.4, 0.7)
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed=seed)

    # Create scenario data for two clusters
    n_nodes = 2 * n_nodes_per_cluster

    # Parameters for Gaussian distribution around each center
    cluster_std_lat = cluster_size_std  # Standard deviation in latitude (degrees)
    cluster_std_lon = cluster_size_std  # Standard deviation in longitude (degrees)

    # Generate coordinates for both clusters
    coordinates = []
    node_ids = []

    for cluster_idx, (center_lat, center_lon) in enumerate(cluster_centers):
        # Generate radial distances using Gaussian distribution
        # Convert to polar coordinates for radial distribution
        radial_distances = np.abs(rng.normal(0, 0.2, n_nodes_per_cluster))  # km equivalent
        angles = rng.uniform(0, 2 * np.pi, n_nodes_per_cluster)

        # Convert polar to lat/lon offsets (approximate: 1 degree ≈ 111 km)
        lat_offsets = radial_distances * np.cos(angles) / 111.0
        lon_offsets = radial_distances * np.sin(angles) / 111.0

        # Add some additional Gaussian noise for more realistic distribution
        lat_noise = rng.normal(0, cluster_std_lat, n_nodes_per_cluster)
        lon_noise = rng.normal(0, cluster_std_lon, n_nodes_per_cluster)

        # Calculate final coordinates
        cluster_lats = center_lat + lat_offsets + lat_noise
        cluster_lons = center_lon + lon_offsets + lon_noise

        # Create node IDs for this cluster
        for i in range(n_nodes_per_cluster):
            node_id = f"cluster_{cluster_idx + 1}:node_{i + 1}"
            node_ids.append(node_id)
            coordinates.append((cluster_lats[i], cluster_lons[i]))

    lats, lons = zip(*coordinates, strict=False)

    # Generate population sizes with larger populations near cluster centers
    populations = []
    for i, (lat, lon) in enumerate(coordinates):
        cluster_idx = i // n_nodes_per_cluster
        center_lat, center_lon = cluster_centers[cluster_idx]

        # Calculate distance from center
        distance = np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)

        # Larger populations near center, smaller populations at edges
        # Base population: 50,000 to 200,000
        base_pop = rng.integers(50000, 200000)

        # Distance factor: closer to center = larger population
        distance_factor = np.exp(-distance / 0.1)  # Exponential decay
        final_pop = int(base_pop * (0.3 + 0.7 * distance_factor))

        populations.append(final_pop)

    # Convert to numpy array for compatibility with visualization
    populations = np.array(populations)

    # Generate MCV1 coverage (40% to 70%)
    mcv1_coverage = rng.uniform(mcv1_coverage_range[0], mcv1_coverage_range[1], n_nodes)

    # Create scenario DataFrame
    scenario_data = pl.DataFrame({"id": node_ids, "pop": populations, "lat": lats, "lon": lons, "mcv1": mcv1_coverage})

    return scenario_data


def satellites_scenario(
    core_population: int = 500_000,
    satellite_population: int = 100_000,
    n_towns: int = 30,
    max_distance: float = 200,
    mcv1: float = 0.50,
    seed: int | None = 52,
    population_std: float = 0.3,
):
    """
    Create a cluster of nodes with a single large node in the center (core) surrounded by smaller nodes (satellites).

    Args:
        core_population (int): Population of the core city
        satellite_population (int): Population of the satellite cities
        n_towns (int): Number of towns to create
        max_distance (float): Maximum distance from center (km)
        mcv1 (float): MCV1 coverage
        seed (int): Random seed for reproducibility

    Returns:
        pl.DataFrame: Scenario DataFrame.
    """
    # Create one major city at the center
    towns = []

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed=seed)

    # Major city (largest population)
    towns.append(
        {
            "id": "0",
            "lat": 0.0,
            "lon": 0.0,
            "pop": core_population,  # Large central city
            "mcv1": mcv1,
        }
    )

    # Create smaller towns at various distances
    for i in range(1, n_towns):
        # Random distance from center (weighted toward closer distances)
        # distance = np.random.exponential(scale=max_distance / 3)
        distance = np.sqrt(rng.uniform(0, max_distance**2))
        distance = min(distance, max_distance)

        # Random angle
        angle = rng.uniform(0, 2 * np.pi)

        # Convert to lat/lon (approximate, assuming 1 degree ≈ 111 km)
        lat = (distance * np.cos(angle)) / 111.0
        lon = (distance * np.sin(angle)) / 111.0

        # Population follows power law distribution (many small towns, few large ones)
        pop = satellite_population + rng.integers(0, int(core_population / 10))
        # pop = np.random.lognormal(mean=np.log(target_median), sigma=np.log(target_median) / 10)  # Log-normal distribution
        # pop = max(5000, min(100000, int(pop)))  # Constrain to reasonable range

        # MCV1 coverage varies slightly
        mcv1 = rng.normal(mcv1, 0.05)
        mcv1 = max(0.0, min(0.95, mcv1))

        towns.append({"id": str(i), "lat": lat, "lon": lon, "pop": int(pop), "mcv1": mcv1})

    return pl.DataFrame(towns)
