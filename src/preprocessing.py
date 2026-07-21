import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data")


def load_data() -> pd.DataFrame:
    """Load and merge transactions with products."""
    trans = pd.read_csv(DATA_PATH / "Transactions.csv")
    trans['date'] = pd.to_datetime(trans['Date_and_time_of_unloading']).dt.date
    trans['date'] = pd.to_datetime(trans['date'])
    return trans



def make_daily_series(trans: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to daily level."""
    daily = trans.groupby('date').agg(
        total_sales=('Sale_amount', 'sum'),
        total_profit=('Profit', 'sum'),
        transactions=('Product_code', 'count')
    ).reset_index()

    # usa só 2020 - ano mais completo
    daily = daily[daily['date'].dt.year == 2020].reset_index(drop=True)

    # fill missing dates
    full_range = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(full_range).reset_index()
    daily.columns = ['date'] + list(daily.columns[1:])
    daily = daily.fillna(0)

    # remove dias sem venda
    daily = daily[daily['total_sales'] > 0].reset_index(drop=True)

    # remove outliers extremos
    p99 = daily['total_sales'].quantile(0.99)
    daily = daily[daily['total_sales'] <= p99].reset_index(drop=True)

    return daily


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features for XGBoost."""
    df = df.copy()
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # lag features
    df['lag_7'] = df['total_sales'].shift(7)
    df['lag_14'] = df['total_sales'].shift(14)
    df['lag_30'] = df['total_sales'].shift(30)

    # rolling means
    df['rolling_7'] = df['total_sales'].shift(1).rolling(7).mean()
    df['rolling_30'] = df['total_sales'].shift(1).rolling(30).mean()

    return df.dropna()


if __name__ == "__main__":
    trans = load_data()
    daily = make_daily_series(trans)
    daily_feat = add_time_features(daily)
    print(f"Daily series: {daily.shape}")
    print(f"With features: {daily_feat.shape}")
    print(daily_feat.head())