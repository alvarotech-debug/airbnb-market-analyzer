"""
Data loading utilities for Inside Airbnb datasets.

Handles downloading, decompression, and loading of CSV/GeoJSON files
from the Inside Airbnb public data repository.
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    """Load project configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_latest_data_url(
    city: str = "austin",
    state: str = "tx",
    country: str = "united-states",
    base_url: str = "https://data.insideairbnb.com",
) -> dict[str, str]:
    """Discover the latest available data date for a city on Inside Airbnb.

    Scrapes the Inside Airbnb website to find the most recent data release
    and builds download URLs for all data files.

    Args:
        city: City name (lowercase, hyphenated).
        state: Two-letter state code (lowercase).
        country: Country slug used in Inside Airbnb URLs.
        base_url: Base URL for Inside Airbnb data.

    Returns:
        Dictionary mapping file names to their full download URLs.
    """
    index_url = f"https://insideairbnb.com/get-the-data/"
    logger.info("Fetching data index from %s", index_url)

    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Look for links that match our city's data path
    city_pattern = f"{country}/{state}/{city}"
    date_pattern = re.compile(
        rf"{re.escape(base_url)}/{city_pattern}/(\d{{4}}-\d{{2}}-\d{{2}})/data/"
    )

    dates = set()
    for link in soup.find_all("a", href=True):
        match = date_pattern.search(link["href"])
        if match:
            dates.add(match.group(1))

    if not dates:
        raise ValueError(
            f"No data found for {city}, {state}. "
            f"Check https://insideairbnb.com/get-the-data/ for available cities."
        )

    latest_date = sorted(dates)[-1]
    logger.info("Latest data date for %s: %s", city, latest_date)

    data_base = f"{base_url}/{city_pattern}/{latest_date}"

    return {
        "listings.csv.gz": f"{data_base}/data/listings.csv.gz",
        "calendar.csv.gz": f"{data_base}/data/calendar.csv.gz",
        "reviews.csv.gz": f"{data_base}/data/reviews.csv.gz",
        "neighbourhoods.geojson": f"{data_base}/visualisations/neighbourhoods.geojson",
        "date": latest_date,
    }


def download_file(url: str, destination: Path) -> Path:
    """Download a file from a URL to a local path.

    Args:
        url: Full URL to download.
        destination: Local file path to save to.

    Returns:
        Path to the downloaded file.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s", url)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(destination, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r  {destination.name}: {pct:.0f}%", end="", flush=True)

    print()
    logger.info("Saved to %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
    return destination


def download_all_data(config_path: str | Path = CONFIG_PATH) -> dict[str, Path]:
    """Download all data files for the configured city.

    Args:
        config_path: Path to settings.yaml.

    Returns:
        Dictionary mapping file names to local paths.
    """
    config = load_config(config_path)
    loc = config["location"]
    data_cfg = config["data"]
    raw_dir = Path(config_path).resolve().parent.parent / data_cfg["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if data_cfg.get("manual_date"):
        date = data_cfg["manual_date"]
        base = (
            f"{data_cfg['base_url']}/{loc['country']}/{loc['state']}/{loc['city']}"
            f"/{date}"
        )
        urls = {
            f: f"{base}/data/{f}" for f in data_cfg["files"]
        }
        urls[data_cfg["geojson"]] = f"{base}/visualisations/{data_cfg['geojson']}"
        urls["date"] = date
    else:
        urls = get_latest_data_url(
            city=loc["city"],
            state=loc["state"],
            country=loc["country"],
            base_url=data_cfg["base_url"],
        )

    print(f"Data date: {urls['date']}")
    print(f"Downloading to: {raw_dir}\n")

    paths = {}
    for filename in [*data_cfg["files"], data_cfg["geojson"]]:
        url = urls[filename]
        dest = raw_dir / filename
        download_file(url, dest)
        paths[filename] = dest

    print(f"\nAll files downloaded successfully.")
    return paths


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------

def load_listings(data_path: str | Path) -> pd.DataFrame:
    """Load listings data with appropriate dtypes.

    Args:
        data_path: Path to listings CSV (plain or .gz).

    Returns:
        Raw listings DataFrame (not yet cleaned).
    """
    df = pd.read_csv(
        data_path,
        low_memory=False,
        dtype={
            "id": "int64",
            "host_id": "int64",
        },
    )
    logger.info("Loaded listings: %d rows, %d columns", *df.shape)
    return df


def load_calendar(data_path: str | Path) -> pd.DataFrame:
    """Load calendar data with date parsing.

    Args:
        data_path: Path to calendar CSV (plain or .gz).

    Returns:
        Raw calendar DataFrame.
    """
    df = pd.read_csv(
        data_path,
        parse_dates=["date"],
        dtype={"listing_id": "int64"},
    )
    logger.info("Loaded calendar: %d rows", len(df))
    return df


def load_reviews(data_path: str | Path) -> pd.DataFrame:
    """Load reviews data with date parsing.

    Args:
        data_path: Path to reviews CSV (plain or .gz).

    Returns:
        Raw reviews DataFrame.
    """
    df = pd.read_csv(
        data_path,
        parse_dates=["date"],
        dtype={"listing_id": "int64", "id": "int64", "reviewer_id": "int64"},
    )
    logger.info("Loaded reviews: %d rows", len(df))
    return df


def load_geojson(data_path: str | Path) -> gpd.GeoDataFrame:
    """Load neighborhood boundaries as GeoDataFrame.

    Args:
        data_path: Path to neighbourhoods.geojson.

    Returns:
        GeoDataFrame with neighborhood polygons.
    """
    gdf = gpd.read_file(data_path)
    logger.info("Loaded %d neighborhoods", len(gdf))
    return gdf
