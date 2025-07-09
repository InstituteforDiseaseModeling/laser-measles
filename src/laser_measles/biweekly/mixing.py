import numpy as np
import polars as pl

# from astropy.coordinates import angular_separation

# Constants for gravity diffusion model
MAX_DISTANCE = 100000000  # km, used to prevent self-migration
MIN_DISTANCE = 10  # km, minimum distance to prevent excessive neighbor migration


def pairwise_haversine(df):  # TODO: use angular separation formula instead
    """Pairwise distances for all (lon, lat) points using the Haversine formula.

    Args:
        df (pl.DataFrame): Polars DataFrame with 'lon' and 'lat' columns

    Returns:
        Pairwise distances in kilometers
    """

    # mean earth radius in km
    earth_radius_km = 6367

    # convert from degrees to radians using polars
    data = np.deg2rad(df[["lon", "lat"]].to_numpy())
    lon = data[:, 0]
    lat = data[:, 1]

    # matrices of pairwise differences for latitudes & longitudes
    dlat = lat[:, None] - lat
    dlon = lon[:, None] - lon

    # vectorized haversine distance calculation
    d = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(d))


def init_gravity_diffusion(df: pl.DataFrame, scale: float, dist_exp: float) -> np.ndarray:
    if len(df) == 1:
        return np.ones((1, 1))

    # Calculate pairwise distances
    distances = pairwise_haversine(df)

    pops = np.array(df["pop"])
    pops = pops[:, np.newaxis].T
    pops = np.repeat(pops, pops.size, axis=0).astype(np.float64)

    np.fill_diagonal(distances, 100000000)  # Prevent divide by zero errors and self migration
    diffusion_matrix = pops / (distances + 10) ** dist_exp  # minimum distance prevents excessive neighbor migration
    np.fill_diagonal(diffusion_matrix, 0)

    # normalize average total outbound migration to 1
    diffusion_matrix = diffusion_matrix / np.mean(np.sum(diffusion_matrix, axis=1))

    diffusion_matrix *= scale
    diagonal = 1 - np.sum(diffusion_matrix, axis=1)  # normalized outbound migration by source
    np.fill_diagonal(diffusion_matrix, diagonal)

    return diffusion_matrix


def pairwise_haversine_(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances for all (lon, lat) points using the Haversine formula.

    Args:
        lon: Array of longitude values in degrees
        lat: Array of latitude values in degrees

    Returns:
        Compressed matrix of pairwise distances in kilometers, where only the upper triangle
        (excluding diagonal) is stored. The full matrix can be reconstructed using:
        full_matrix = np.zeros((n, n))
        full_matrix[np.triu_indices(n, k=1)] = compressed_matrix
        full_matrix = full_matrix + full_matrix.T
    """
    earth_radius_km = 6367
    n = len(lon)

    # Get upper triangle indices (excluding diagonal)
    i, j = np.triu_indices(n, k=1)

    # Calculate differences only for the upper triangle
    dlat = lat[j] - lat[i]
    dlon = lon[j] - lon[i]

    # Calculate cosines only for the needed pairs
    cos_lat_i = np.cos(lat[i])
    cos_lat_j = np.cos(lat[j])

    # vectorized haversine distance calculation for upper triangle only
    d = np.sin(dlat / 2) ** 2 + cos_lat_i * cos_lat_j * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(d))


def init_gravity_diffusion_(df: pl.DataFrame | tuple[np.ndarray, np.ndarray], scale: float, dist_exp: float) -> np.ndarray:
    """Initialize a gravity diffusion matrix for population mixing.

    Args:
        df: Either a DataFrame with 'population', 'lat', and 'lon' columns,
            or a tuple of (lon, lat) arrays
        scale: Scaling factor for the diffusion matrix
        dist_exp: Distance exponent for the gravity model

    Returns:
        Normalized diffusion matrix where each row sums to 1
    """
    if len(df) == 1:
        return np.ones((1, 1))

    n = len(df)
    compressed_distances = pairwise_haversine(df["lon"].to_numpy(), df["lat"].to_numpy())
    pops = df["pop"].to_numpy()

    # Get the indices for the upper triangle
    i, j = np.triu_indices(n, k=1)

    # Calculate population products only for the upper triangle
    pop_products = pops[i] * pops[j]

    # Calculate diffusion values for upper triangle
    diffusion_values = pop_products / (compressed_distances + MIN_DISTANCE) ** dist_exp

    # Create the full matrix
    diffusion_matrix = np.zeros((n, n))
    diffusion_matrix[i, j] = diffusion_values
    diffusion_matrix[j, i] = diffusion_values  # Mirror the values

    # Calculate row sums for normalization
    row_sums = np.sum(diffusion_matrix, axis=1)
    mean_sum = np.mean(row_sums)

    # Normalize and scale
    diffusion_matrix = diffusion_matrix / mean_sum * scale

    # Calculate and set diagonal values
    diagonal = 1 - np.sum(diffusion_matrix, axis=1)
    np.fill_diagonal(diffusion_matrix, diagonal)

    return diffusion_matrix
