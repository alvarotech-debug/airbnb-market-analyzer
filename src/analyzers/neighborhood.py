"""
Neighborhood-level analysis for Airbnb listings.

Provides geographic market segmentation: ADR by area, listing density,
saturation scores, and GeoDataFrame preparation for choropleth maps.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd


class NeighborhoodAnalyzer:
    """Analyze market dynamics at the neighborhood level."""

    def __init__(
        self,
        listings: pd.DataFrame,
        geo_df: Optional[gpd.GeoDataFrame] = None,
        config: dict | None = None,
    ):
        self.df = listings
        self.geo_df = geo_df
        self.config = config or {}

    def get_adr_by_neighborhood(self, min_listings: int = 3) -> pd.DataFrame:
        """Average Daily Rate by neighborhood, filtered by minimum listing count."""
        nh = (
            self.df.groupby("neighbourhood_cleansed")["price"]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={
                "mean": "adr_mean", "median": "adr_median",
                "std": "adr_std", "count": "listings",
            })
        )
        nh = nh[nh["listings"] >= min_listings].sort_values("adr_mean", ascending=False)
        return nh

    def get_listing_density(self) -> pd.DataFrame:
        """Listing count per neighborhood."""
        counts = (
            self.df["neighbourhood_cleansed"]
            .value_counts()
            .rename_axis("neighbourhood")
            .reset_index(name="listing_count")
        )
        counts["pct_of_total"] = counts["listing_count"] / counts["listing_count"].sum() * 100
        return counts

    def get_market_saturation_score(self) -> pd.DataFrame:
        """Composite saturation score combining supply and availability.

        Score = normalized(listing_count) + normalized(1 - avg_availability_rate).
        Higher score → more saturated market.
        """
        nh = self.df.groupby("neighbourhood_cleansed").agg(
            listing_count=("id", "count"),
            avg_availability_365=("availability_365", "mean"),
        )

        # Normalize 0-1
        nh["supply_score"] = (
            (nh["listing_count"] - nh["listing_count"].min())
            / (nh["listing_count"].max() - nh["listing_count"].min() + 1e-9)
        )
        nh["demand_score"] = 1 - (
            (nh["avg_availability_365"] - nh["avg_availability_365"].min())
            / (nh["avg_availability_365"].max() - nh["avg_availability_365"].min() + 1e-9)
        )
        nh["saturation_score"] = (nh["supply_score"] + nh["demand_score"]) / 2

        return nh.sort_values("saturation_score", ascending=False)

    def get_price_heatmap_data(self) -> gpd.GeoDataFrame:
        """Merge neighborhood-level stats with GeoJSON for choropleth mapping."""
        if self.geo_df is None:
            raise ValueError("GeoJSON data required for heatmap.")

        adr = self.get_adr_by_neighborhood(min_listings=1).reset_index()
        geo = self.geo_df.copy()

        # Inside Airbnb GeoJSON uses 'neighbourhood' as column name
        merge_col = "neighbourhood"
        if merge_col not in geo.columns:
            merge_col = geo.columns[0]

        # Ensure matching types (GeoJSON may store zip codes as str)
        adr["neighbourhood_cleansed"] = adr["neighbourhood_cleansed"].astype(str)
        geo[merge_col] = geo[merge_col].astype(str)

        merged = geo.merge(
            adr,
            left_on=merge_col,
            right_on="neighbourhood_cleansed",
            how="left",
        )
        merged["avg_price"] = merged["adr_mean"]
        return merged

    def get_neighborhood_profile(self, neighborhood: str) -> dict:
        """Deep dive metrics for a single neighborhood."""
        nh_df = self.df[self.df["neighbourhood_cleansed"] == neighborhood]
        if nh_df.empty:
            return {"error": f"No listings found in '{neighborhood}'"}

        return {
            "neighborhood": neighborhood,
            "total_listings": len(nh_df),
            "adr_mean": nh_df["price"].mean(),
            "adr_median": nh_df["price"].median(),
            "dominant_room_type": nh_df["room_type"].mode().iloc[0] if not nh_df["room_type"].mode().empty else None,
            "avg_rating": nh_df["review_scores_rating"].mean(),
            "pct_superhost": nh_df["is_superhost"].mean() * 100 if "is_superhost" in nh_df.columns else None,
            "avg_accommodates": nh_df["accommodates"].mean(),
        }
