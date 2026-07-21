import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)


def evaluate(y_true, y_pred, model_name: str) -> dict:
    """Calculates RMSE, MAE and R2."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Results:")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  MAE:  {mae:,.2f}")
    print(f"  R2:   {r2:.4f}")
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima(daily: pd.DataFrame) -> tuple:
    """Train SARIMA model."""
    print("\nTraining SARIMA...")

    series = daily.set_index('date')['total_sales']

    # train/test split — last 60 days
    train = series.iloc[:-60]
    test = series.iloc[-60:]

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)

    y_pred = result.forecast(steps=60)
    metrics = evaluate(test.values, y_pred.values, 'SARIMA')

    with open(MODELS_PATH / "sarima_model.pkl", "wb") as f:
        pickle.dump(result, f)

    print("  SARIMA model saved!")
    return result, y_pred, test, metrics


def train_xgboost(daily_feat: pd.DataFrame) -> tuple:
    """Train XGBoost model with time series features."""
    print("\nTraining XGBoost...")

    feature_cols = [
        'dayofweek', 'month', 'quarter', 'year',
        'dayofyear', 'is_weekend',
        'lag_7', 'lag_14', 'lag_30',
        'rolling_7', 'rolling_30'
    ]

    X = daily_feat[feature_cols]
    y = daily_feat['total_sales']

    # log transform pra reduzir variabilidade
    y_log = np.log1p(y)

    # time series split — last 60 rows as test
    train_size = len(daily_feat) - 60
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y_log.iloc[:train_size], y_log.iloc[train_size:]
    y_test_orig = np.expm1(y_test)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    metrics = evaluate(y_test_orig, y_pred, 'XGBoost')

    # save
    with open(MODELS_PATH / "xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 5 features:")
    print(importance.head())

    print("  XGBoost model saved!")
    return model, y_pred, y_test_orig, metrics, importance


if __name__ == "__main__":
    from src.preprocessing import load_data, make_daily_series, add_time_features

    trans = load_data()
    daily = make_daily_series(trans)
    daily_feat = add_time_features(daily)

    sarima_result, y_pred_sarima, test_sarima, sarima_metrics = train_sarima(daily)
    xgb_model, y_pred, y_test, xgb_metrics, importance = train_xgboost(daily_feat)

    print("\n=== Model Comparison ===")
   #print(f"Prophet RMSE: {prophet_metrics['rmse']:,.2f} | R2: {prophet_metrics['r2']:.4f}")
    print(f"XGBoost RMSE: {xgb_metrics['rmse']:,.2f} | R2: {xgb_metrics['r2']:.4f}")