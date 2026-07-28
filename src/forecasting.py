import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from neuralprophet import NeuralProphet

MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)

FEATURE_COLS = [
    'dayofweek', 'month', 'quarter', 'year',
    'dayofyear', 'is_weekend', 'is_monday',
    'lag_7', 'lag_14', 'lag_30',
    'lag_price_7', 'lag_promo_7', 'lag_stock_7',
    'rolling_7', 'rolling_14', 'rolling_30', 'rolling_std_7',
]


def evaluate(y_true, y_pred, model_name: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  RMSE: {rmse:,.2f} | MAE: {mae:,.2f} | R2: {r2:.4f}")
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


def to_pandas(df_pl: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars to Pandas for sklearn/statsmodels."""
    return df_pl.to_pandas()


def train_sarima(df_pl: pl.DataFrame, sku: str) -> dict:
    print(f"\n  [SARIMA]")
    df = to_pandas(df_pl)
    series = df.set_index('date')['units_sold']
    train, test = series.iloc[:-60], series.iloc[-60:]

    model = SARIMAX(
        train, order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    y_pred = result.forecast(steps=60)
    metrics = evaluate(test.values, y_pred.values, 'SARIMA')

    with open(MODELS_PATH / f"sarima_{sku}.pkl", "wb") as f:
        pickle.dump(result, f)
    return metrics


def cross_validate(model, X, y_log, n_splits=3) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y_log.iloc[tr])
        y_pred = np.expm1(model.predict(X.iloc[te]))
        scores.append(r2_score(np.expm1(y_log.iloc[te]), y_pred))
    return float(np.mean(scores))


def train_xgboost(df_pl: pl.DataFrame, sku: str) -> dict:
    print(f"\n  [XGBoost]")
    df = to_pandas(df_pl)
    X = df[FEATURE_COLS]
    y = df['units_sold']
    y_log = np.log1p(y)

    cv_score = cross_validate(
        xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        X, pd.Series(y_log)
    )
    print(f"  CV R2: {cv_score:.4f}")

    train_size = len(df) - 60
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X.iloc[:train_size], y_log.iloc[:train_size],
              eval_set=[(X.iloc[train_size:], y_log.iloc[train_size:])],
              verbose=False)

    y_pred = np.expm1(model.predict(X.iloc[train_size:]))
    y_test = np.expm1(y_log.iloc[train_size:])
    metrics = evaluate(y_test, y_pred, 'XGBoost')

    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"  Top feature: {importance.iloc[0]['feature']} ({importance.iloc[0]['importance']:.3f})")

    with open(MODELS_PATH / f"xgboost_{sku}.pkl", "wb") as f:
        pickle.dump(model, f)

    return metrics, importance


def train_lightgbm(df_pl: pl.DataFrame, sku: str) -> dict:
    print(f"\n  [LightGBM]")
    df = to_pandas(df_pl)
    X = df[FEATURE_COLS]
    y = df['units_sold']
    y_log = np.log1p(y)

    cv_score = cross_validate(
        lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        X, pd.Series(y_log)
    )
    print(f"  CV R2: {cv_score:.4f}")

    train_size = len(df) - 60
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    model.fit(
        X.iloc[:train_size], y_log.iloc[:train_size],
        eval_set=[(X.iloc[train_size:], y_log.iloc[train_size:])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)]
    )

    y_pred = np.expm1(model.predict(X.iloc[train_size:]))
    y_test = np.expm1(y_log.iloc[train_size:])
    metrics = evaluate(y_test, y_pred, 'LightGBM')

    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"  Top feature: {importance.iloc[0]['feature']} ({importance.iloc[0]['importance']})")

    with open(MODELS_PATH / f"lightgbm_{sku}.pkl", "wb") as f:
        pickle.dump(model, f)

    return metrics, importance


def train_neuralprophet(df_pl: pl.DataFrame, sku: str) -> dict:
    print(f"\n  [NeuralProphet]")
    df = to_pandas(df_pl)[['date', 'units_sold', 'promotion_flag', 'avg_price']].copy()
    df = df.rename(columns={'date': 'ds', 'units_sold': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    train = df.iloc[:-60].copy()
    test = df.iloc[-60:].copy()

    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_lags=14,
        n_forecasts=1,
        learning_rate=0.01,
        epochs=50,
    )
    model.add_lagged_regressor("promotion_flag")
    model.add_lagged_regressor("avg_price")

    model.fit(train, freq="D", progress="none")

    # usa o dataframe de teste diretamente como future
    future = test.copy()
    forecast = model.predict(future)

    y_pred = forecast['yhat1'].dropna().values
    y_true = test['y'].values[:len(y_pred)]

    metrics = evaluate(y_true, y_pred, 'NeuralProphet')

    with open(MODELS_PATH / f"neuralprophet_{sku}.pkl", "wb") as f:
        pickle.dump(model, f)

    return metrics


if __name__ == "__main__":
    from src.preprocessing import load_top_skus_dataset

    print("Loading datasets...")
    datasets = load_top_skus_dataset(n=5)

    all_results = {}

    for sku, df_pl in datasets.items():
        print(f"\n{'='*50}")
        print(f"SKU: {sku} | {len(df_pl)} samples")
        print(f"{'='*50}")

        results = {}
        results['sarima'] = train_sarima(df_pl, sku)
        results['xgboost'], _ = train_xgboost(df_pl, sku)
        results['lightgbm'], _ = train_lightgbm(df_pl, sku)
        results['neuralprophet'] = train_neuralprophet(df_pl, sku)

        all_results[sku] = results

        print(f"\n  === {sku} Summary ===")
        for model_name, metrics in results.items():
            print(f"  {model_name:15} RMSE: {metrics['rmse']:,.2f} | R2: {metrics['r2']:.4f}")

    print(f"\n{'='*50}")
    print("FINAL RESULTS — ALL SKUs")
    print(f"{'='*50}")
    for sku, results in all_results.items():
        best = min(results.values(), key=lambda x: x['rmse'])
        print(f"{sku}: best model = {best['model']} (RMSE: {best['rmse']:,.2f} | R2: {best['r2']:.4f})")