import polars as pl
import numpy as np
from pathlib import Path

DATA_PATH = Path("data")


def load_data() -> pl.DataFrame:
    """Load FMCG dataset with Polars."""
    df = pl.read_csv(DATA_PATH / "FMCG_2022_2024.csv")
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    return df


def get_top_skus(df: pl.DataFrame, n: int = 5) -> list:
    """Get top N SKUs by total units sold — Pareto approach."""
    top = (
        df.group_by("sku")
        .agg(pl.col("units_sold").sum().alias("total_sold"))
        .sort("total_sold", descending=True)
        .head(n)
    )
    skus = top["sku"].to_list()
    totals = top["total_sold"].to_list()
    print("Top SKUs by volume:")
    for sku, total in zip(skus, totals):
        print(f"  {sku}: {total:,} units")
    return skus


def make_daily_series(df: pl.DataFrame, sku: str) -> pl.DataFrame:
    """
    Aggregate daily sales for a specific SKU.
    Includes price, promotion and stock features.
    """
    daily = (
        df.filter(pl.col("sku") == sku)
        .group_by("date")
        .agg([
            pl.col("units_sold").sum().alias("units_sold"),
            pl.col("price_unit").mean().alias("avg_price"),
            pl.col("promotion_flag").max().alias("promotion_flag"),
            pl.col("stock_available").sum().alias("stock_available"),
            pl.col("delivered_qty").sum().alias("delivered_qty"),
        ])
        .sort("date")
    )

    # fill missing dates
    date_range = pl.date_range(
        daily["date"].min(),
        daily["date"].max(),
        interval="1d",
        eager=True
    )
    full = pl.DataFrame({"date": date_range})
    daily = full.join(daily, on="date", how="left").fill_null(0)

    return daily


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add temporal and lag features using Polars."""
    df = df.with_columns([
        pl.col("date").dt.weekday().alias("dayofweek"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.quarter().alias("quarter"),
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.ordinal_day().alias("dayofyear"),
        (pl.col("date").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        (pl.col("date").dt.weekday() == 0).cast(pl.Int8).alias("is_monday"),
    ])

    # lag features
    df = df.with_columns([
        pl.col("units_sold").shift(7).alias("lag_7"),
        pl.col("units_sold").shift(14).alias("lag_14"),
        pl.col("units_sold").shift(30).alias("lag_30"),
        pl.col("avg_price").shift(7).alias("lag_price_7"),
        pl.col("promotion_flag").shift(7).alias("lag_promo_7"),
        pl.col("stock_available").shift(7).alias("lag_stock_7"),
    ])

    # rolling means
    df = df.with_columns([
        pl.col("units_sold").shift(1).rolling_mean(7).alias("rolling_7"),
        pl.col("units_sold").shift(1).rolling_mean(14).alias("rolling_14"),
        pl.col("units_sold").shift(1).rolling_mean(30).alias("rolling_30"),
        pl.col("units_sold").shift(1).rolling_std(7).alias("rolling_std_7"),
    ])

    return df.drop_nulls()


def load_sku_dataset(sku: str) -> pl.DataFrame:
    """Full pipeline for a single SKU."""
    df = load_data()
    daily = make_daily_series(df, sku)
    daily_feat = add_time_features(daily)
    return daily_feat


def load_top_skus_dataset(n: int = 5) -> dict:
    """Load dataset for top N SKUs — returns dict {sku: DataFrame}."""
    df = load_data()
    top_skus = get_top_skus(df, n)

    datasets = {}
    for sku in top_skus:
        daily = make_daily_series(df, sku)
        daily_feat = add_time_features(daily)
        datasets[sku] = daily_feat
        print(f"  {sku}: {len(daily_feat)} samples after feature engineering")

    return datasets


if __name__ == "__main__":
    print("Loading FMCG dataset...")
    df = load_data()
    print(f"Shape: {df.shape}")

    print("\nTop SKUs:")
    top_skus = get_top_skus(df)

    print("\nBuilding datasets per SKU...")
    datasets = load_top_skus_dataset()

    print("\nSample — first SKU:")
    first_sku = list(datasets.keys())[0]
    print(datasets[first_sku].head())