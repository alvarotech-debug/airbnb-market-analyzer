"""
Professional chart generation for Airbnb market analysis.

Provides a consistent visual style across all analysis outputs with
export capabilities for presentations and reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from ..data_loader import CONFIG_PATH, load_config

# Segment color mapping
SEGMENT_COLORS = {
    "Budget": "#2ecc71",
    "Mid-Range": "#3498db",
    "Premium": "#e74c3c",
}

ROOM_TYPE_COLORS = {
    "Entire home/apt": "#3498db",
    "Private room": "#2ecc71",
    "Shared room": "#f39c12",
    "Hotel room": "#9b59b6",
}


class ChartGenerator:
    """Creates styled, publication-ready charts for STR market analysis."""

    def __init__(self, config_path: str | Path = CONFIG_PATH):
        self.config = load_config(config_path)
        self.viz = self.config["visualization"]
        self.output_dir = Path(
            config_path
        ).resolve().parent.parent / self.config["output"]["charts"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._apply_style()

    def _apply_style(self):
        """Set global matplotlib/seaborn style."""
        try:
            plt.style.use(self.viz["style"])
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid")

        sns.set_palette(self.viz["palette"])
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "axes.titlesize": self.viz["font_title"],
            "axes.labelsize": self.viz["font_label"],
            "xtick.labelsize": self.viz["font_tick"],
            "ytick.labelsize": self.viz["font_tick"],
            "figure.dpi": 100,
            "savefig.dpi": self.viz["dpi"],
            "savefig.bbox": "tight",
        })

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        """Save figure to outputs/charts/ and return the path."""
        path = self.output_dir / f"{filename}.{self.viz['format']}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        return path

    def _add_source(self, ax: plt.Axes):
        """Add data source attribution to bottom of chart."""
        ax.annotate(
            "Data: Inside Airbnb (insideairbnb.com)",
            xy=(1, -0.12), xycoords="axes fraction",
            ha="right", fontsize=8, color="#95a5a6", style="italic",
        )

    # ------------------------------------------------------------------
    # Market Overview Charts
    # ------------------------------------------------------------------

    def plot_price_distribution(
        self,
        df: pd.DataFrame,
        price_col: str = "price",
        title: str = "Nightly Price Distribution",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Histogram with KDE of listing prices."""
        fig, ax = plt.subplots(figsize=self.viz["figsize_default"])

        sns.histplot(
            df[price_col].dropna(), bins=50, kde=True,
            color=self.viz["colors"]["secondary"], alpha=0.7, ax=ax,
        )

        median_price = df[price_col].median()
        ax.axvline(median_price, color=self.viz["colors"]["accent"], linestyle="--",
                    linewidth=2, label=f"Median: ${median_price:,.0f}")

        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_xlabel("Price per Night (USD)")
        ax.set_ylabel("Number of Listings")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend()
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    def plot_adr_by_room_type(
        self,
        df: pd.DataFrame,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Bar chart of Average Daily Rate by room type."""
        adr = df.groupby("room_type")["price"].agg(["mean", "median", "count"])
        adr = adr.sort_values("mean", ascending=True)

        colors = [ROOM_TYPE_COLORS.get(rt, "#bdc3c7") for rt in adr.index]

        fig, ax = plt.subplots(figsize=self.viz["figsize_default"])
        bars = ax.barh(adr.index, adr["mean"], color=colors, edgecolor="white", height=0.6)

        for bar, (_, row) in zip(bars, adr.iterrows()):
            ax.text(
                bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f"${row['mean']:,.0f}  (n={row['count']:,.0f})",
                va="center", fontsize=11, fontweight="bold",
            )

        ax.set_title("Average Daily Rate by Room Type", fontweight="bold", pad=15)
        ax.set_xlabel("Average Price per Night (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    def plot_supply_breakdown(
        self,
        df: pd.DataFrame,
        col: str = "room_type",
        title: str = "Supply by Room Type",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Donut chart of listing supply breakdown."""
        counts = df[col].value_counts()
        colors = [ROOM_TYPE_COLORS.get(rt, "#bdc3c7") for rt in counts.index]

        fig, ax = plt.subplots(figsize=self.viz["figsize_square"][:1] * 2)
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90, pctdistance=0.82,
            wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
        )
        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight("bold")

        ax.set_title(title, fontweight="bold", pad=20, fontsize=self.viz["font_title"])
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    def plot_rating_distribution(
        self,
        df: pd.DataFrame,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Histogram of review scores."""
        scores = df["review_scores_rating"].dropna()
        fig, ax = plt.subplots(figsize=self.viz["figsize_default"])

        sns.histplot(scores, bins=20, color=self.viz["colors"]["secondary"],
                     alpha=0.7, ax=ax, kde=True)

        mean_score = scores.mean()
        ax.axvline(mean_score, color=self.viz["colors"]["accent"], linestyle="--",
                    linewidth=2, label=f"Mean: {mean_score:.2f}")

        ax.set_title("Distribution of Review Scores", fontweight="bold", pad=15)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of Listings")
        ax.legend()
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    # ------------------------------------------------------------------
    # Seasonality Charts
    # ------------------------------------------------------------------

    def plot_price_seasonality(
        self,
        monthly_data: pd.DataFrame,
        price_col: str = "avg_price",
        month_col: str = "month",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Line chart of monthly average price with confidence band."""
        fig, ax = plt.subplots(figsize=self.viz["figsize_wide"])

        months = monthly_data[month_col]
        prices = monthly_data[price_col]
        month_labels = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        ax.plot(months, prices, marker="o", linewidth=2.5,
                color=self.viz["colors"]["secondary"], markersize=8, zorder=3)

        if "std_price" in monthly_data.columns:
            std = monthly_data["std_price"]
            ax.fill_between(months, prices - std, prices + std,
                            alpha=0.15, color=self.viz["colors"]["secondary"])

        # Highlight high season
        high_months = self.config["analysis"]["high_season_months"]
        for m in months:
            if m in high_months:
                ax.axvspan(m - 0.4, m + 0.4, alpha=0.06, color="#e74c3c")

        ax.set_title("Average Nightly Price by Month", fontweight="bold", pad=15)
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Price (USD)")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    def plot_occupancy_by_month(
        self,
        monthly_data: pd.DataFrame,
        occ_col: str = "occupancy_rate",
        month_col: str = "month",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Bar chart of estimated occupancy rate by month."""
        fig, ax = plt.subplots(figsize=self.viz["figsize_wide"])
        month_labels = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        high_months = self.config["analysis"]["high_season_months"]
        colors = [
            self.viz["colors"]["accent"] if m in high_months
            else self.viz["colors"]["secondary"]
            for m in monthly_data[month_col]
        ]

        ax.bar(monthly_data[month_col], monthly_data[occ_col] * 100,
               color=colors, edgecolor="white", width=0.7)

        ax.set_title("Estimated Occupancy Rate by Month", fontweight="bold", pad=15)
        ax.set_xlabel("Month")
        ax.set_ylabel("Occupancy Rate (%)")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    # ------------------------------------------------------------------
    # Neighborhood Charts
    # ------------------------------------------------------------------

    def plot_adr_by_neighborhood(
        self,
        df: pd.DataFrame,
        top_n: int = 10,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Horizontal bar chart of top neighborhoods by ADR."""
        adr = (
            df.groupby("neighbourhood_cleansed")["price"]
            .agg(["mean", "count"])
            .query("count >= 3")
            .nlargest(top_n, "mean")
            .sort_values("mean")
        )

        fig, ax = plt.subplots(figsize=self.viz["figsize_default"])
        bars = ax.barh(
            adr.index, adr["mean"],
            color=self.viz["colors"]["secondary"], edgecolor="white", height=0.6,
        )

        for bar, (_, row) in zip(bars, adr.iterrows()):
            ax.text(
                bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"${row['mean']:,.0f} (n={row['count']:.0f})",
                va="center", fontsize=10,
            )

        ax.set_title(f"Top {top_n} Neighborhoods by Average Price", fontweight="bold", pad=15)
        ax.set_xlabel("Average Price per Night (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    def plot_price_heatmap(
        self,
        geo_df: gpd.GeoDataFrame,
        price_col: str = "avg_price",
        title: str = "Average Nightly Price by Neighborhood",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Choropleth map of prices by neighborhood."""
        fig, ax = plt.subplots(figsize=self.viz["figsize_square"])

        geo_df.plot(
            column=price_col,
            cmap="YlOrRd",
            linewidth=0.8,
            edgecolor="white",
            legend=True,
            legend_kwds={"label": "Avg Price (USD)", "shrink": 0.6},
            ax=ax,
            missing_kwds={"color": "#f0f0f0", "label": "No data"},
        )

        ax.set_title(title, fontweight="bold", pad=15, fontsize=self.viz["font_title"])
        ax.set_axis_off()
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    def plot_listing_density(
        self,
        df: pd.DataFrame,
        top_n: int = 15,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Bar chart of listing count by neighborhood."""
        counts = df["neighbourhood_cleansed"].value_counts().head(top_n).sort_values()

        fig, ax = plt.subplots(figsize=self.viz["figsize_default"])
        ax.barh(counts.index, counts.values,
                color=self.viz["colors"]["secondary"], edgecolor="white", height=0.6)

        for i, (nh, count) in enumerate(counts.items()):
            ax.text(count + 1, i, str(count), va="center", fontsize=10)

        ax.set_title(f"Top {top_n} Neighborhoods by Listing Count", fontweight="bold", pad=15)
        ax.set_xlabel("Number of Listings")
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    # ------------------------------------------------------------------
    # Competitive Charts
    # ------------------------------------------------------------------

    def plot_price_vs_rating(
        self,
        df: pd.DataFrame,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Scatter plot of price vs review rating, colored by segment."""
        plot_df = df.dropna(subset=["price", "review_scores_rating"]).copy()

        fig, ax = plt.subplots(figsize=self.viz["figsize_default"])

        if "segment" in plot_df.columns:
            for seg, color in SEGMENT_COLORS.items():
                mask = plot_df["segment"] == seg
                if mask.any():
                    ax.scatter(
                        plot_df.loc[mask, "review_scores_rating"],
                        plot_df.loc[mask, "price"],
                        c=color, label=seg, alpha=0.6, s=50, edgecolors="white",
                    )
        else:
            ax.scatter(
                plot_df["review_scores_rating"], plot_df["price"],
                c=self.viz["colors"]["secondary"], alpha=0.6, s=50, edgecolors="white",
            )

        ax.set_title("Price vs. Review Rating", fontweight="bold", pad=15)
        ax.set_xlabel("Review Score (Rating)")
        ax.set_ylabel("Price per Night (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(title="Segment")
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    def plot_segment_distribution(
        self,
        df: pd.DataFrame,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Bar chart of listing count by price segment."""
        if "segment" not in df.columns:
            return plt.figure()

        counts = df["segment"].value_counts().reindex(SEGMENT_COLORS.keys()).fillna(0)
        colors = [SEGMENT_COLORS[s] for s in counts.index]

        fig, ax = plt.subplots(figsize=self.viz["figsize_default"])
        bars = ax.bar(counts.index, counts.values, color=colors,
                      edgecolor="white", width=0.5)

        total = counts.sum()
        for bar, count in zip(bars, counts.values):
            pct = count / total * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{count:.0f}\n({pct:.0f}%)", ha="center", fontsize=11, fontweight="bold")

        ax.set_title("Market Segmentation by Price", fontweight="bold", pad=15)
        ax.set_ylabel("Number of Listings")
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    def plot_amenity_frequency(
        self,
        amenity_counts: pd.Series,
        top_n: int = 15,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Horizontal bar chart of most common amenities."""
        top = amenity_counts.head(top_n).sort_values()

        fig, ax = plt.subplots(figsize=self.viz["figsize_default"])
        ax.barh(top.index, top.values,
                color=self.viz["colors"]["secondary"], edgecolor="white", height=0.6)

        for i, (amenity, count) in enumerate(top.items()):
            ax.text(count + 0.5, i, str(count), va="center", fontsize=10)

        ax.set_title(f"Top {top_n} Most Common Amenities", fontweight="bold", pad=15)
        ax.set_xlabel("Number of Listings")
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig

    def plot_boxplot_by_group(
        self,
        df: pd.DataFrame,
        x: str,
        y: str = "price",
        title: str = "Price Distribution",
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Box plot of a numeric column grouped by a category."""
        fig, ax = plt.subplots(figsize=self.viz["figsize_default"])

        order = df.groupby(x)[y].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=x, y=y, order=order, palette="muted",
                    showfliers=False, ax=ax)

        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_ylabel("Price per Night (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.xticks(rotation=25, ha="right")
        self._add_source(ax)
        fig.tight_layout()

        if save_as:
            self._save(fig, save_as)
        return fig
