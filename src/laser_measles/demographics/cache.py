import shutil
from pathlib import Path

import diskcache
from appdirs import user_cache_dir


def load_cache(cache_dir: str | None = None) -> diskcache.Cache:
    """
    Load the cache for the application.

    Args:
        cache_dir: The cache directory to use. If None, the default cache directory will be used.

    Returns:
        The cache object.
    """
    resolve_cache_dir = cache_dir or get_cache_dir()
    return diskcache.Cache(resolve_cache_dir)


def get_cache_dir() -> str:
    """Get the cache directory for the application."""
    return user_cache_dir("laser_measles")


def clear_cache(cache_dir: str | None = None) -> None:
    """Clear diskcache data from the application cache."""
    # Clear diskcache
    with load_cache(cache_dir) as cache:
        cache.clear()


def clear_all_cache() -> None:
    """Clear all cached data from the application cache."""
    clear_cache()
    cache_dir = get_cache_dir()
    if Path(cache_dir).exists():
        shutil.rmtree(cache_dir)


def clear_cache_dir(dir: str) -> None:
    """Clear all cached data from a specific directory."""
    resolve_dir = Path(get_cache_dir()) / dir
    if resolve_dir.exists():
        shutil.rmtree(resolve_dir)

def get_all_cache_keys() -> list[str]:
    """Get all the cache keys for the application."""
    with load_cache() as c:
        return list(c.iterkeys())
