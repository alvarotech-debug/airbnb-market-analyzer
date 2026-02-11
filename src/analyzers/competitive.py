"""
Competitive landscape analysis for Airbnb listings.

Segments the market by price tier, analyzes amenity differentiation,
identifies gaps, and computes superhost premiums.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


class CompetitiveAnalyzer:
    """Analyze competitive dynamics and market positioning."""

    def __init__(self, listings: pd.DataFrame, config: dict):
        self.df = listings
        self.config = config
        self._segments = config["analysis"]["segments"]

    def _assign_segment(self, price: float) -> str:
        """Map a price to its segment label."""
        for name, bounds in self._segments.items():
            if bounds["min"] <= price < bounds["max"]:
                return bounds["label"]
        return "Other"

    def segment_by_price(self) -> pd.DataFrame:
        """Assign each listing to a price segment and return summary."""
        df = self.df.copy()
        df["segment"] = df["price"].apply(self._assign_segment)
        summary = (
            df.groupby("segment")
            .agg(
                listings=("id", "count"),
                avg_price=("price", "mean"),
                median_price=("price", "median"),
                avg_rating=("review_scores_rating", "mean"),
            )
        )
        summary["pct"] = summary["listings"] / summary["listings"].sum() * 100

        # Reindex to keep segment order
        order = [s["label"] for s in self._segments.values()]
        summary = summary.reindex([o for o in order if o in summary.index])
        return summary

    def add_segments_to_df(self) -> pd.DataFrame:
        """Return listings DataFrame with a 'segment' column added."""
        df = self.df.copy()
        df["segment"] = df["price"].apply(self._assign_segment)
        return df

    def get_price_vs_rating(self) -> pd.DataFrame:
        """Subset for scatter analysis: price, rating, segment."""
        df = self.add_segments_to_df()
        return df[["price", "review_scores_rating", "segment", "room_type",
                    "neighbourhood_cleansed"]].dropna(
            subset=["price", "review_scores_rating"]
        )

    def get_amenity_analysis(self) -> dict:
        """Frequency and segment breakdown of amenities."""
        if "amenities_list" not in self.df.columns:
            return {"error": "amenities_list column missing — run data_cleaner first"}

        # Overall frequency
        all_amenities = []
        for lst in self.df["amenities_list"]:
            all_amenities.extend(lst)
        overall = pd.Series(Counter(all_amenities)).sort_values(ascending=False)

        # By segment
        df = self.add_segments_to_df()
        by_segment = {}
        for seg_label in [s["label"] for s in self._segments.values()]:
            seg_df = df[df["segment"] == seg_label]
            seg_amenities = []
            for lst in seg_df["amenities_list"]:
                seg_amenities.extend(lst)
            by_segment[seg_label] = (
                pd.Series(Counter(seg_amenities)).sort_values(ascending=False).head(20)
            )

        return {
            "overall": overall,
            "by_segment": by_segment,
            "total_unique": len(overall),
        }

    def get_superhost_premium(self) -> dict:
        """Compare superhost vs regular host metrics."""
        if "is_superhost" not in self.df.columns:
            return {}

        groups = self.df.groupby("is_superhost").agg(
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            avg_rating=("review_scores_rating", "mean"),
            avg_reviews=("number_of_reviews", "mean"),
            count=("id", "count"),
        )

        if True not in groups.index or False not in groups.index:
            return {"note": "Insufficient data for superhost comparison"}

        sh = groups.loc[True]
        reg = groups.loc[False]
        return {
            "superhost_avg_price": sh["avg_price"],
            "regular_avg_price": reg["avg_price"],
            "price_premium_pct": (sh["avg_price"] - reg["avg_price"]) / reg["avg_price"] * 100,
            "superhost_avg_rating": sh["avg_rating"],
            "regular_avg_rating": reg["avg_rating"],
            "superhost_count": int(sh["count"]),
            "regular_count": int(reg["count"]),
        }

    def identify_market_gaps(self) -> list[dict]:
        """Identify underserved market segments.

        Looks for combinations of neighborhood × room_type × segment
        with low supply relative to overall demand signals.
        """
        df = self.add_segments_to_df()
        gaps = []

        # Find neighborhoods with few premium listings
        top_nh = self.config["analysis"]["top_neighborhoods"]
        nh_counts = df["neighbourhood_cleansed"].value_counts().head(top_nh).index

        for nh in nh_counts:
            nh_df = df[df["neighbourhood_cleansed"] == nh]
            seg_counts = nh_df["segment"].value_counts()
            total = len(nh_df)

            for seg_label in [s["label"] for s in self._segments.values()]:
                count = seg_counts.get(seg_label, 0)
                pct = count / total * 100 if total else 0
                if count < 3 or pct < 10:
                    gaps.append({
                        "neighbourhood": nh,
                        "segment": seg_label,
                        "current_listings": count,
                        "pct_of_neighborhood": round(pct, 1),
                        "opportunity": f"Low {seg_label.lower()} supply in {nh}",
                    })

        return sorted(gaps, key=lambda g: g["current_listings"])[:15]
