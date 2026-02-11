"""
Seasonality analysis for Airbnb calendar data.

Identifies temporal patterns in pricing and occupancy, including
monthly trends, weekend premiums, and peak season identification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalityAnalyzer:
    """Analyze temporal patterns in pricing and availability."""

    def __init__(
        self,
        listings: pd.DataFrame,
        calendar: pd.DataFrame,
        config: dict,
    ):
        self.listings = listings
        self.calendar = calendar
        self.config = config

    def get_price_by_month(self) -> pd.DataFrame:
        """Average price aggregated by month across all listings."""
        monthly = (
            self.calendar.dropna(subset=["price"])
            .groupby("month")["price"]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={
                "mean": "avg_price", "median": "median_price",
                "std": "std_price", "count": "observations",
            })
            .reset_index()
        )
        return monthly

    def get_availability_by_month(self) -> pd.DataFrame:
        """Estimated occupancy rate by month (1 - availability_rate)."""
        monthly = (
            self.calendar.groupby("month")["available"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "availability_rate", "count": "observations"})
            .reset_index()
        )
        monthly["occupancy_rate"] = 1 - monthly["availability_rate"]
        return monthly

    def identify_peak_season(self) -> dict:
        """Classify months into high/low season and compute premiums."""
        price_by_month = self.get_price_by_month()
        high_months = self.config["analysis"]["high_season_months"]
        low_months = self.config["analysis"]["low_season_months"]

        high_avg = price_by_month[
            price_by_month["month"].isin(high_months)
        ]["avg_price"].mean()
        low_avg = price_by_month[
            price_by_month["month"].isin(low_months)
        ]["avg_price"].mean()

        premium_pct = ((high_avg - low_avg) / low_avg * 100) if low_avg else 0

        return {
            "high_season_months": high_months,
            "low_season_months": low_months,
            "high_season_avg_price": high_avg,
            "low_season_avg_price": low_avg,
            "seasonal_premium_pct": premium_pct,
        }

    def get_weekend_vs_weekday_pricing(self) -> dict:
        """Compare weekend (Fri-Sat) vs weekday pricing."""
        cal = self.calendar.dropna(subset=["price"])
        weekend = cal[cal["is_weekend"]]["price"]
        weekday = cal[~cal["is_weekend"]]["price"]

        weekend_avg = weekend.mean()
        weekday_avg = weekday.mean()
        premium = ((weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg else 0

        return {
            "weekend_avg": weekend_avg,
            "weekday_avg": weekday_avg,
            "weekend_premium_pct": premium,
            "weekend_median": weekend.median(),
            "weekday_median": weekday.median(),
        }

    def get_price_by_day_of_week(self) -> pd.DataFrame:
        """Average price for each day of the week."""
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = (
            self.calendar.dropna(subset=["price"])
            .groupby("day_of_week")["price"]
            .agg(["mean", "median"])
            .rename(columns={"mean": "avg_price", "median": "median_price"})
            .reset_index()
        )
        daily["day_name"] = daily["day_of_week"].map(
            {i: name for i, name in enumerate(day_names)}
        )
        return daily
