import numpy as np
import polars as pl


def two_cluster_scenario(
    seed: int = 42,
    n_nodes_per_cluster: int = 50,
    cluster_centers: list[tuple[float, float]] | None = None,
    cluster_size_std: float = 0.3,
    mcv1_coverage_range: tuple[float, float] | None = None,
):
    """ Generate a synthetic scenario with two clusters of nodes.

    Args:
        seed: Random seed for reproducibility.
        n_nodes_per_cluster: Number of nodes per cluster.
        cluster_centers: List of tuples representing the centers of the clusters.
        cluster_size_std: Standard deviation of the Gaussian distribution for cluster size.
        mcv1_coverage_range: Range of MCV1 coverage percentages.
    """

    # Set defaults for mutable arguments
    if cluster_centers is None:
        cluster_centers = [(40.0, 4.0), (34.0, 10.0)]
    if mcv1_coverage_range is None:
        mcv1_coverage_range = (0.4, 0.7)
    # Set random seed for reproducibility
    np.random.seed(seed)

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
        radial_distances = np.abs(np.random.normal(0, 0.2, n_nodes_per_cluster))  # km equivalent
        angles = np.random.uniform(0, 2 * np.pi, n_nodes_per_cluster)

        # Convert polar to lat/lon offsets (approximate: 1 degree ≈ 111 km)
        lat_offsets = radial_distances * np.cos(angles) / 111.0
        lon_offsets = radial_distances * np.sin(angles) / 111.0

        # Add some additional Gaussian noise for more realistic distribution
        lat_noise = np.random.normal(0, cluster_std_lat, n_nodes_per_cluster)
        lon_noise = np.random.normal(0, cluster_std_lon, n_nodes_per_cluster)

        # Calculate final coordinates
        cluster_lats = center_lat + lat_offsets + lat_noise
        cluster_lons = center_lon + lon_offsets + lon_noise

        # Create node IDs for this cluster
        for i in range(n_nodes_per_cluster):
            node_id = f"cluster_{cluster_idx + 1}:node_{i + 1}"
            node_ids.append(node_id)
            coordinates.append((cluster_lats[i], cluster_lons[i]))

    lats, lons = zip(*coordinates)

    # Generate population sizes with larger populations near cluster centers
    populations = []
    for i, (lat, lon) in enumerate(coordinates):
        cluster_idx = i // n_nodes_per_cluster
        center_lat, center_lon = cluster_centers[cluster_idx]

        # Calculate distance from center
        distance = np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)

        # Larger populations near center, smaller populations at edges
        # Base population: 50,000 to 200,000
        base_pop = np.random.randint(50000, 200000)

        # Distance factor: closer to center = larger population
        distance_factor = np.exp(-distance / 0.1)  # Exponential decay
        final_pop = int(base_pop * (0.3 + 0.7 * distance_factor))

        populations.append(final_pop)

    # Convert to numpy array for compatibility with visualization
    populations = np.array(populations)

    # Generate MCV1 coverage (40% to 70%)
    mcv1_coverage = np.random.uniform(mcv1_coverage_range[0], mcv1_coverage_range[1], n_nodes)

    # Create scenario DataFrame
    scenario_data = pl.DataFrame({"id": node_ids, "pop": populations, "lat": lats, "lon": lons, "mcv1": mcv1_coverage})

    return scenario_data
