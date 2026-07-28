# Beverage Demand Forecasting

> Multi-model time series forecasting pipeline for FMCG beverage SKUs, with interpretability analysis via SHAP and seasonal decomposition.

---

## Problem Statement

A beverage distributor operating across multiple regions and retail channels faced a recurring challenge: demand planning was done manually, based on historical averages and intuition. This approach had three critical failure points:

- **Stockouts and overstock**: without accurate SKU-level forecasts, some products ran out during peak periods while others accumulated excess inventory — both generating direct financial loss
- **Promotion blindness**: promotional events were not systematically incorporated into planning, making it impossible to anticipate demand spikes or post-promotion dips
- **No seasonality awareness**: demand patterns varied significantly by day of week, month, and season — but planners lacked a structured way to quantify and act on these patterns

The business needed a data-driven forecasting solution that could operate at the SKU level, incorporate promotional and pricing signals, and provide interpretable outputs that category managers could actually use.

---

## Proposed Solution

An end-to-end demand forecasting pipeline that:

- Identifies the highest-priority SKUs using a **Pareto/ABC analysis** — focusing modeling effort where it matters most
- Builds **daily time series** per SKU with rich temporal and business features
- Trains and compares **four forecasting models** — SARIMA, XGBoost, LightGBM, and NeuralProphet — selecting the best per SKU
- Provides **interpretability** appropriate to each model type: SHAP for tree-based models, seasonal decomposition for statistical and neural models
- Surfaces actionable insights for category managers: which SKUs are in decline, which have strong weekly patterns, and when seasonal peaks occur

---

## Architecture

```
FMCG_2022_2024.csv (190k rows)
        |
        v
Polars preprocessing pipeline
        |
        v
Pareto analysis — Top 5 SKUs by volume
        |
        v
Daily aggregation per SKU
(units_sold, avg_price, promotion_flag, stock_available)
        |
        v
Feature engineering
(lag features, rolling means, temporal features)
        |
        v
Model training — 4 models per SKU
    SARIMA | XGBoost | LightGBM | NeuralProphet
        |
        v
Best model selection per SKU (RMSE)
        |
        v
Interpretability
    Tree-based -> SHAP
    SARIMA     -> Seasonal Decomposition
    NeuralProphet -> Component Decomposition
```

---

## Dataset

- **Source**: [FMCG Daily Sales Data 2022-2024](https://www.kaggle.com/datasets/beatafaron/fmcg-daily-sales-data-to-2022-2024) — Kaggle, CC0
- **Records**: 190,757 transactions
- **Period**: January 2022 — December 2024
- **SKUs**: 30 products across 5 categories
- **Categories**: Milk, Yogurt, ReadyMeal, Juice, SnackBar
- **Channels**: Retail, Discount, E-commerce
- **Regions**: PL-Central, PL-North, PL-South
- **Key features**: units_sold, price_unit, promotion_flag, stock_available, delivered_qty

---

## Top SKUs — Pareto Analysis

| SKU | Category | Total Units Sold | Samples |
|-----|----------|-----------------|---------|
| YO-029 | Yogurt | 177,798 | 1,008 |
| YO-005 | Yogurt | 176,095 | 995 |
| YO-012 | Yogurt | 170,575 | 979 |
| MI-026 | Milk | 152,324 | 1,045 |
| RE-004 | ReadyMeal | 149,559 | 960 |

---

## Model Results

| SKU | Best Model | RMSE | R2 |
|-----|-----------|------|-----|
| YO-029 | SARIMA | 29.85 | -0.005 |
| YO-005 | LightGBM | 29.51 | 0.044 |
| YO-012 | SARIMA | 30.03 | -0.105 |
| MI-026 | NeuralProphet | 27.20 | 0.064 |
| RE-004 | NeuralProphet | 40.94 | 0.153 |

No single model dominated across all SKUs — each product has a distinct demand pattern that favors different modeling approaches. This is consistent with real-world FMCG forecasting, where SKU heterogeneity is the norm.

---

## Interpretability

### SHAP — YO-005 (LightGBM)
Rolling averages (30-day and 14-day) are the dominant features, followed by day-of-year — indicating strong long-term trend and annual seasonality. Weekly patterns and promotional signals have limited impact on this SKU.

### Seasonal Decomposition — YO-029 and YO-012 (SARIMA)
Both SKUs show a clear demand cycle: high in early 2022, declining through 2023, recovering in 2024. Weekly seasonality is stable but low-amplitude. Residuals show no systematic patterns, confirming the model captured the main structure.

### NeuralProphet Components — MI-026 and RE-004
- **MI-026 (Milk)**: consistent downward trend from 2022 to 2024 — a market decline signal. Bi-annual seasonality with peaks in spring and autumn.
- **RE-004 (ReadyMeal)**: strong weekly seasonality (±40 units) — likely weekend demand peaks. Clear summer peak in yearly seasonality. General downward trend with partial recovery in 2024.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Data processing | Polars, Pandas, NumPy |
| Time series models | SARIMA (statsmodels), XGBoost, LightGBM, NeuralProphet |
| Interpretability | SHAP, seasonal decomposition |
| Visualization | Matplotlib |
| Environment | Python 3.10, venv |

---

## Project Structure

```
beverage-demand-forecast/
├── src/
│   ├── preprocessing.py    # Polars pipeline, Pareto, feature engineering
│   └── forecasting.py      # SARIMA, XGBoost, LightGBM, NeuralProphet
├── data/                   # Raw data (not tracked)
├── models/                 # Saved models and plots (not tracked)
├── notebooks/              # Exploratory analysis
└── requirements.txt
```

---

## Running Locally

```bash
git clone https://github.com/elemcarames/beverage-demand-forecast
cd beverage-demand-forecast
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle datasets download -d beatafaron/fmcg-daily-sales-data-to-2022-2024 -p data/ --unzip

# Run preprocessing
python -m src.preprocessing

# Run forecasting + interpretability
python -m src.forecasting
```

---

## Key Insights

- **No universal winner**: the best model varies by SKU — SARIMA works well for stable weekly patterns, NeuralProphet captures complex non-linear seasonality, LightGBM excels when lag features carry predictive power
- **MI-026 in decline**: consistent downward trend over 3 years — warrants strategic review of shelf allocation and pricing
- **RE-004 weekend effect**: strong weekly seasonality suggests replenishment should be differentiated by day of week
- **Promotion signal weak**: promotion_flag has low SHAP importance across SKUs — either promotions are infrequent or their effect is already captured by price features

---

## Next Steps

- [ ] Dash dashboard with interactive forecast visualization per SKU
- [ ] Price elasticity analysis — quantify promotion and price impact on demand
- [ ] SKU-level Champion/Challenger retraining pipeline
- [ ] Expand to all 30 SKUs with automated model selection

---

*Part of the DS portfolio of [Elem Tamirys dos Santos Carames](https://elemcarames.github.io)*
