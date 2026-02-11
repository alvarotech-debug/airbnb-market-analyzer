"""
Market overview analysis for Airbnb listings.

Provides high-level metrics: ADR, supply breakdown, market concentration,
host rankings, and rating distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketOverviewAnalyzer:
    """Compute aggregate market metrics from cleaned listings data."""

    def __init__(self, listings: pd.DataFrame, config: dict):
        self.df = listings
        self.config = config

    def get_total_active_listings(self) -> int:
        return len(self.df)

    def get_adr_by_room_type(self) -> pd.DataFrame:
        """Average Daily Rate grouped by room type."""
        return (
            self.df.groupby("room_type")["price"]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={"mean": "adr_mean", "median": "adr_median",
                             "std": "adr_std", "count": "listings"})
            .sort_values("adr_mean", ascending=False)
        )

    def get_price_distribution(self) -> dict:
        """Descriptive statistics for listing prices."""
        prices = self.df["price"].dropna()
        return {
            "mean": prices.mean(),
            "median": prices.median(),
            "std": prices.std(),
            "min": prices.min(),
            "max": prices.max(),
            "q25": prices.quantile(0.25),
            "q75": prices.quantile(0.75),
            "count": len(prices),
        }

    def get_supply_by_property_type(self) -> pd.DataFrame:
        """Listing count and percentage by property type."""
        counts = self.df["property_type"].value_counts()
        pct = counts / counts.sum() * 100
        return pd.DataFrame({"count": counts, "pct": pct}).head(15)

    def get_supply_by_room_type(self) -> pd.DataFrame:
        """Listing count and percentage by room type."""
        counts = self.df["room_type"].value_counts()
        pct = counts / counts.sum() * 100
        return pd.DataFrame({"count": counts, "pct": pct})

    def get_top_hosts(self, n: int = 20) -> pd.DataFrame:
        """Top hosts ranked by number of listings."""
        host_cols = ["host_id", "host_name"]
        existing = [c for c in host_cols if c in self.df.columns]
        if not existing:
            return pd.DataFrame()
        hosts = (
            self.df.groupby(existing)
            .agg(
                listings=("id", "count"),
                avg_price=("price", "mean"),
                avg_rating=("review_scores_rating", "mean"),
            )
            .nlargest(n, "listings")
            .reset_index()
        )
        hosts["market_share_pct"] = hosts["listings"] / len(self.df) * 100
        return hosts

    def get_market_concentration(self) -> dict:
        """Herfindahl–Hirschman Index (HHI) based on host listing shares."""
        shares = self.df["host_id"].value_counts() / len(self.df) * 100
        hhi = (shares ** 2).sum()
        top_10_share = shares.nlargest(10).sum()
        return {
            "hhi": hhi,
            "top_10_host_share_pct": top_10_share,
            "unique_hosts": self.df["host_id"].nunique(),
        }

    def get_rating_distribution(self) -> dict:
        """Statistics for review scores."""
        scores = self.df["review_scores_rating"].dropna()
        return {
            "mean": scores.mean(),
            "median": scores.median(),
            "std": scores.std(),
            "pct_above_4": (scores >= 4.0).mean() * 100,
            "pct_above_4_5": (scores >= 4.5).mean() * 100,
            "total_with_rating": len(scores),
            "total_listings": len(self.df),
        }

    def get_summary(self) -> dict:
        """Executive summary of the market."""
        price_dist = self.get_price_distribution()
        concentration = self.get_market_concentration()
        ratings = self.get_rating_distribution()
        return {
            "total_listings": self.get_total_active_listings(),
            "adr_mean": price_dist["mean"],
            "adr_median": price_dist["median"],
            "unique_hosts": concentration["unique_hosts"],
            "top_10_host_share": concentration["top_10_host_share_pct"],
            "avg_rating": ratings["mean"],
            "pct_superhost": (
                self.df["is_superhost"].mean() * 100
                if "is_superhost" in self.df.columns else None
            ),
        }
