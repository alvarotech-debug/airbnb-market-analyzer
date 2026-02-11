"""
Data cleaning and normalization for Inside Airbnb datasets.

Transforms raw CSV data into analysis-ready DataFrames by handling
missing values, normalizing types, filtering outliers, and adding
computed fields.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data_loader import CONFIG_PATH, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def clean_price(price_str) -> float:
    """Convert a price string like '$1,234.56' to a float.

    Returns NaN for missing or unparseable values.
    """
    if pd.isna(price_str):
        return np.nan
    s = str(price_str).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Amenities
# ---------------------------------------------------------------------------

def parse_amenities(amenities_str) -> list[str]:
    """Parse an amenities column value into a Python list.

    Inside Airbnb stores amenities as a JSON array string, e.g.
    '["Wifi", "Kitchen", "Air conditioning"]'.
    """
    if pd.isna(amenities_str):
        return []
    try:
        parsed = json.loads(amenities_str)
        if isinstance(parsed, list):
            return [str(a).strip() for a in parsed if a]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Listings cleaning pipeline
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate listings by id, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate listings", removed)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop rows with missing critical values.

    Strategy:
      - bedrooms/beds: fill with median grouped by room_type
      - review scores: fill with NaN (kept for analysis, not dropped)
      - Drop rows missing id, price, or coordinates
    """
    critical = ["id", "price", "latitude", "longitude"]
    before = len(df)
    df = df.dropna(subset=[c for c in critical if c in df.columns])
    logger.info("Dropped %d rows missing critical fields", before - len(df))

    for col in ["bedrooms", "beds"]:
        if col in df.columns:
            medians = df.groupby("room_type")[col].transform("median")
            df[col] = df[col].fillna(medians).fillna(1)

    return df


def filter_active_listings(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep only listings that appear to be actively rented.

    Uses cleaning thresholds from config to determine activity.
    """
    clean_cfg = config["cleaning"]
    before = len(df)

    # Price bounds
    price_min = clean_cfg["price"]["min"]
    price_max = clean_cfg["price"]["max"]
    df = df[(df["price"] >= price_min) & (df["price"] <= price_max)]
    logger.info("After price filter ($%d-$%d): %d rows", price_min, price_max, len(df))

    # Activity: must have reviews OR availability
    min_reviews = clean_cfg["min_reviews"]
    has_reviews = df["number_of_reviews"] >= min_reviews
    has_availability = df.get("availability_365", pd.Series(dtype=float)) > 0
    df = df[has_reviews | has_availability]

    logger.info(
        "Filtered %d → %d active listings", before, len(df)
    )
    return df


def add_calculated_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns useful for analysis."""
    # Price per person
    if "accommodates" in df.columns:
        df["price_per_person"] = df["price"] / df["accommodates"].replace(0, np.nan)

    # Estimated monthly revenue (price × booked days in 30)
    if "availability_30" in df.columns:
        df["est_booked_30"] = 30 - df["availability_30"]
        df["est_monthly_revenue"] = df["price"] * df["est_booked_30"]

    # Review recency
    if "last_review" in df.columns:
        today = pd.Timestamp.now().normalize()
        df["last_review_dt"] = pd.to_datetime(df["last_review"], errors="coerce")
        df["days_since_review"] = (today - df["last_review_dt"]).dt.days

    # Boolean helpers
    if "host_is_superhost" in df.columns:
        df["is_superhost"] = df["host_is_superhost"].map({"t": True, "f": False})

    if "instant_bookable" in df.columns:
        df["is_instant_bookable"] = df["instant_bookable"].map({"t": True, "f": False})

    return df


def clean_listings(
    df: pd.DataFrame,
    config_path: str | Path = CONFIG_PATH,
) -> pd.DataFrame:
    """Master cleaning pipeline for listings data.

    Applies all cleaning steps in order and returns a ready-to-analyze
    DataFrame. The original DataFrame is not modified.

    Args:
        df: Raw listings DataFrame.
        config_path: Path to settings.yaml.

    Returns:
        Cleaned listings DataFrame with normalized types and added fields.
    """
    config = load_config(config_path)
    df = df.copy()
    logger.info("Starting listings cleaning (%d rows, %d cols)", *df.shape)

    # 1. Remove duplicates
    df = remove_duplicates(df)

    # 2. Normalize price
    df["price"] = df["price"].apply(clean_price)

    # 3. Handle missing values (after price normalization)
    df = handle_missing_values(df)

    # 4. Filter active listings
    df = filter_active_listings(df, config)

    # 5. Add calculated fields
    df = add_calculated_fields(df)

    # 6. Parse amenities
    if "amenities" in df.columns:
        df["amenities_list"] = df["amenities"].apply(parse_amenities)
        df["amenity_count"] = df["amenities_list"].apply(len)

    logger.info("Cleaning complete: %d rows, %d cols", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Calendar cleaning
# ---------------------------------------------------------------------------

def clean_calendar(
    df: pd.DataFrame,
    config_path: str | Path = CONFIG_PATH,
) -> pd.DataFrame:
    """Clean and normalize calendar data.

    Args:
        df: Raw calendar DataFrame.
        config_path: Path to settings.yaml.

    Returns:
        Cleaned calendar DataFrame with proper types.
    """
    df = df.copy()
    logger.info("Starting calendar cleaning (%d rows)", len(df))

    # Parse dates if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Clean price
    df["price"] = df["price"].apply(clean_price)

    # Convert available to boolean
    df["available"] = df["available"].map({"t": True, "f": False})

    # Add time features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = df["day_of_week"].isin([4, 5])  # Fri, Sat

    # Drop rows with no date
    df = df.dropna(subset=["date"])

    logger.info("Calendar cleaning complete: %d rows", len(df))
    return df
