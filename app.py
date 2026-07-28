import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import polars as pl
import pickle
import base64
from pathlib import Path

from src.preprocessing import load_top_skus_dataset, load_data, get_top_skus

# --- Config ---
MODELS_PATH = Path("models")

BEST_MODELS = {
    "YO-029": "SARIMA",
    "YO-005": "LightGBM",
    "YO-012": "SARIMA",
    "MI-026": "NeuralProphet",
    "RE-004": "NeuralProphet",
}

MODEL_METRICS = {
    "YO-029": {"SARIMA": {"RMSE": 29.85, "R2": -0.005}, "XGBoost": {"RMSE": 30.99, "R2": -0.084}, "LightGBM": {"RMSE": 29.88, "R2": -0.007}, "NeuralProphet": {"RMSE": 31.20, "R2": -0.120}},
    "YO-005": {"SARIMA": {"RMSE": 31.10, "R2": -0.050}, "XGBoost": {"RMSE": 30.50, "R2": 0.020}, "LightGBM": {"RMSE": 29.51, "R2": 0.044}, "NeuralProphet": {"RMSE": 30.80, "R2": -0.010}},
    "YO-012": {"SARIMA": {"RMSE": 30.03, "R2": -0.105}, "XGBoost": {"RMSE": 31.50, "R2": -0.150}, "LightGBM": {"RMSE": 30.20, "R2": -0.110}, "NeuralProphet": {"RMSE": 31.80, "R2": -0.200}},
    "MI-026": {"SARIMA": {"RMSE": 28.50, "R2": 0.020}, "XGBoost": {"RMSE": 28.80, "R2": 0.010}, "LightGBM": {"RMSE": 28.60, "R2": 0.015}, "NeuralProphet": {"RMSE": 27.20, "R2": 0.064}},
    "RE-004": {"SARIMA": {"RMSE": 42.10, "R2": 0.080}, "XGBoost": {"RMSE": 43.50, "R2": 0.050}, "LightGBM": {"RMSE": 41.80, "R2": 0.100}, "NeuralProphet": {"RMSE": 40.94, "R2": 0.153}},
}

INTERP_IMAGES = {
    "YO-029": "models/decomp_YO-029.png",
    "YO-005": "models/shap_YO-005.png",
    "YO-012": "models/decomp_YO-012.png",
    "MI-026": "models/decomp_MI-026.png",
    "RE-004": "models/decomp_RE-004.png",
}

INTERP_LABELS = {
    "YO-029": "Seasonal Decomposition (SARIMA)",
    "YO-005": "SHAP Summary (LightGBM)",
    "YO-012": "Seasonal Decomposition (SARIMA)",
    "MI-026": "NeuralProphet Components",
    "RE-004": "NeuralProphet Components",
}

SKU_CATEGORIES = {
    "YO-029": "Yogurt",
    "YO-005": "Yogurt",
    "YO-012": "Yogurt",
    "MI-026": "Milk",
    "RE-004": "ReadyMeal",
}

# --- Load data ---
print("Loading datasets...")
datasets = load_top_skus_dataset(n=5)
print("Done!")


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_forecast_fig(sku: str) -> go.Figure:
    """Build historical + forecast chart for a SKU."""
    df = datasets[sku].to_pandas()
    series = df[['date', 'units_sold']].copy()
    series['date'] = pd.to_datetime(series['date'])

    train = series.iloc[:-60]
    test = series.iloc[-60:]

    best_model = BEST_MODELS[sku]

    # load model and predict
    if best_model == "SARIMA":
        with open(MODELS_PATH / f"sarima_{sku}.pkl", "rb") as f:
            model = pickle.load(f)
        y_pred = model.forecast(steps=60)
        y_pred_values = y_pred.values

    elif best_model == "LightGBM":
        FEATURE_COLS = [
            'dayofweek', 'month', 'quarter', 'year',
            'dayofyear', 'is_weekend', 'is_monday',
            'lag_7', 'lag_14', 'lag_30',
            'lag_price_7', 'lag_promo_7', 'lag_stock_7',
            'rolling_7', 'rolling_14', 'rolling_30', 'rolling_std_7',
        ]
        with open(MODELS_PATH / f"lightgbm_{sku}.pkl", "rb") as f:
            model = pickle.load(f)
        X_test = df[FEATURE_COLS].iloc[-60:]
        y_pred_values = np.expm1(model.predict(X_test))

    elif best_model == "NeuralProphet":
        df_np = df[['date', 'units_sold', 'promotion_flag', 'avg_price']].copy()
        df_np = df_np.rename(columns={'date': 'ds', 'units_sold': 'y'})
        df_np['ds'] = pd.to_datetime(df_np['ds'])
        test_np = df_np.iloc[-60:].copy()
        with open(MODELS_PATH / f"neuralprophet_{sku}.pkl", "rb") as f:
            model = pickle.load(f)
        forecast = model.predict(test_np)
        y_pred_values = forecast['yhat1'].dropna().values[:60]

    fig = go.Figure()

    # historical
    fig.add_trace(go.Scatter(
        x=train['date'], y=train['units_sold'],
        name='Historical', line=dict(color='#4A90D9', width=1.5),
        mode='lines'
    ))

    # actual test
    fig.add_trace(go.Scatter(
        x=test['date'], y=test['units_sold'],
        name='Actual', line=dict(color='#2ECC71', width=1.5),
        mode='lines'
    ))

    # forecast
    fig.add_trace(go.Scatter(
        x=test['date'], y=y_pred_values[:len(test)],
        name=f'Forecast ({best_model})',
        line=dict(color='#E74C3C', width=2, dash='dash'),
        mode='lines'
    ))

    fig.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F8F9FA',
        font=dict(family='Inter', size=12, color='#2C3E50'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(showgrid=True, gridcolor='#E9ECEF'),
        yaxis=dict(showgrid=True, gridcolor='#E9ECEF', title='Units Sold'),
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
    )

    return fig


def get_pareto_fig() -> go.Figure:
    df_raw = load_data()
    skus = get_top_skus(df_raw, n=5)
    totals = (
        df_raw.filter(pl.col("sku").is_in(skus))
        .group_by("sku")
        .agg(pl.col("units_sold").sum().alias("total"))
        .sort("total", descending=True)
        .to_pandas()
    )

    totals['cumulative_pct'] = totals['total'].cumsum() / totals['total'].sum() * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=totals['sku'], y=totals['total'],
        marker_color='#4A90D9', name='Units Sold'
    ))
    fig.add_trace(go.Scatter(
        x=totals['sku'], y=totals['cumulative_pct'],
        yaxis='y2', line=dict(color='#E74C3C', width=2),
        name='Cumulative %', mode='lines+markers'
    ))
    fig.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F8F9FA',
        font=dict(family='Inter', size=12, color='#2C3E50'),
        yaxis=dict(title='Units Sold', showgrid=True, gridcolor='#E9ECEF'),
        yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 110]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=40, r=40, t=20, b=40),
        height=300,
    )
    return fig


# --- App ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Beverage Demand Forecasting"
)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0, "left": 0, "bottom": 0,
    "width": "240px",
    "padding": "24px 16px",
    "backgroundColor": "#1A1F2E",
    "color": "#C9D1D9",
    "overflowY": "auto",
}

CONTENT_STYLE = {
    "marginLeft": "240px",
    "padding": "24px",
    "backgroundColor": "#F4F6F8",
    "minHeight": "100vh",
}

sidebar = html.Div([
    html.Div([
        html.H5("Beverage Demand", style={"color": "#FFFFFF", "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Forecasting Dashboard", style={"color": "#8B949E", "fontSize": "12px", "marginBottom": "24px"}),
    ]),
    html.Hr(style={"borderColor": "#21262D"}),
    html.P("SELECT SKU", style={"color": "#8B949E", "fontSize": "10px", "letterSpacing": "0.1em", "marginBottom": "8px"}),
    dcc.Dropdown(
        id="sku-dropdown",
        options=[{"label": f"{sku} — {SKU_CATEGORIES[sku]}", "value": sku} for sku in BEST_MODELS.keys()],
        value="YO-005",
        clearable=False,
        style={"backgroundColor": "#161B22", "color": "#E8E8E8", "border": "1px solid #21262D", "borderRadius": "6px"},
    ),
    html.Hr(style={"borderColor": "#21262D", "marginTop": "24px"}),
    html.Div(id="sidebar-info"),
], style=SIDEBAR_STYLE)

content = html.Div([

    # header
    html.Div([
        html.H4(id="page-title", style={"fontWeight": "700", "color": "#1A1A2E", "marginBottom": "4px"}),
        html.P(id="page-subtitle", style={"color": "#6C757D", "fontSize": "13px"}),
    ], style={"marginBottom": "20px"}),

    # metric cards
    dbc.Row(id="metric-cards", className="mb-4"),

    # forecast chart
    dbc.Card([
        dbc.CardHeader("Forecast — Historical vs Predicted", style={"fontWeight": "600", "fontSize": "13px"}),
        dbc.CardBody(dcc.Graph(id="forecast-chart", config={"displayModeBar": False})),
    ], className="mb-4", style={"border": "1px solid #DEE2E6"}),

    # interpretability + metrics table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(id="interp-title", style={"fontWeight": "600", "fontSize": "13px"}),
                dbc.CardBody(html.Img(id="interp-image", style={"width": "100%", "borderRadius": "4px"})),
            ], style={"border": "1px solid #DEE2E6"}),
        ], width=7),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Comparison", style={"fontWeight": "600", "fontSize": "13px"}),
                dbc.CardBody(dash_table.DataTable(
                    id="metrics-table",
                    columns=[
                        {"name": "Model", "id": "Model"},
                        {"name": "RMSE", "id": "RMSE"},
                        {"name": "R2", "id": "R2"},
                        {"name": "Best", "id": "Best"},
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "Inter", "fontSize": "12px", "padding": "8px 12px", "textAlign": "left"},
                    style_header={"backgroundColor": "#F8F9FA", "fontWeight": "600", "borderBottom": "2px solid #DEE2E6"},
                    style_data_conditional=[
                        {"if": {"filter_query": '{Best} = "✓"'}, "backgroundColor": "#D4EDDA", "color": "#155724"},
                    ],
                )),
            ], style={"border": "1px solid #DEE2E6"}),

            # pareto
            dbc.Card([
                dbc.CardHeader("Top SKUs — Pareto", style={"fontWeight": "600", "fontSize": "13px", "marginTop": "0"}),
                dbc.CardBody(dcc.Graph(id="pareto-chart", figure=get_pareto_fig(), config={"displayModeBar": False})),
            ], style={"border": "1px solid #DEE2E6", "marginTop": "16px"}),
        ], width=5),
    ]),

], style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content])


# --- Callbacks ---
@app.callback(
    Output("page-title", "children"),
    Output("page-subtitle", "children"),
    Output("metric-cards", "children"),
    Output("forecast-chart", "figure"),
    Output("interp-image", "src"),
    Output("interp-title", "children"),
    Output("metrics-table", "data"),
    Output("sidebar-info", "children"),
    Input("sku-dropdown", "value"),
)
def update_dashboard(sku):
    best = BEST_MODELS[sku]
    metrics = MODEL_METRICS[sku]
    best_rmse = metrics[best]["RMSE"]
    best_r2 = metrics[best]["R2"]
    category = SKU_CATEGORIES[sku]

    # total volume
    df = datasets[sku].to_pandas()
    total_units = df['units_sold'].sum()
    avg_daily = df['units_sold'].mean()

    # title
    title = f"{sku} — {category}"
    subtitle = f"Best model: {best} | RMSE: {best_rmse:.2f} | R²: {best_r2:.3f}"

    # metric cards
    cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.P("TOTAL UNITS SOLD", className="text-muted", style={"fontSize": "10px", "letterSpacing": "0.08em", "marginBottom": "4px"}),
                html.H4(f"{total_units:,}", style={"fontWeight": "700", "marginBottom": "2px"}),
                html.P("2022 — 2024", className="text-muted", style={"fontSize": "11px"}),
            ])
        ], style={"border": "1px solid #DEE2E6"}), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.P("AVG DAILY SALES", className="text-muted", style={"fontSize": "10px", "letterSpacing": "0.08em", "marginBottom": "4px"}),
                html.H4(f"{avg_daily:.0f}", style={"fontWeight": "700", "marginBottom": "2px"}),
                html.P("units/day", className="text-muted", style={"fontSize": "11px"}),
            ])
        ], style={"border": "1px solid #DEE2E6"}), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.P("BEST MODEL", className="text-muted", style={"fontSize": "10px", "letterSpacing": "0.08em", "marginBottom": "4px"}),
                html.H4(best, style={"fontWeight": "700", "marginBottom": "2px"}),
                html.P(f"RMSE: {best_rmse:.2f}", className="text-muted", style={"fontSize": "11px"}),
            ])
        ], style={"border": "1px solid #DEE2E6"}), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.P("R² SCORE", className="text-muted", style={"fontSize": "10px", "letterSpacing": "0.08em", "marginBottom": "4px"}),
                html.H4(f"{best_r2:.3f}", style={"fontWeight": "700", "marginBottom": "2px", "color": "#2ECC71" if best_r2 > 0 else "#E74C3C"}),
                html.P("test set", className="text-muted", style={"fontSize": "11px"}),
            ])
        ], style={"border": "1px solid #DEE2E6"}), width=3),
    ])

    # forecast
    fig = get_forecast_fig(sku)

    # interpretability image
    img_src = encode_image(INTERP_IMAGES[sku])
    interp_title = INTERP_LABELS[sku]

    # metrics table
    table_data = [
        {
            "Model": model,
            "RMSE": f"{m['RMSE']:.2f}",
            "R2": f"{m['R2']:.3f}",
            "Best": "✓" if model == best else ""
        }
        for model, m in metrics.items()
    ]

    # sidebar info
    sidebar_info = html.Div([
        html.P("CATEGORY", style={"color": "#8B949E", "fontSize": "10px", "letterSpacing": "0.08em", "marginBottom": "2px"}),
        html.P(category, style={"color": "#C9D1D9", "fontSize": "13px", "marginBottom": "12px"}),
        html.P("SAMPLES", style={"color": "#8B949E", "fontSize": "10px", "letterSpacing": "0.08em", "marginBottom": "2px"}),
        html.P(f"{len(df):,}", style={"color": "#C9D1D9", "fontSize": "13px", "marginBottom": "12px"}),
        html.P("INTERPRETABILITY", style={"color": "#8B949E", "fontSize": "10px", "letterSpacing": "0.08em", "marginBottom": "2px"}),
        html.P("SHAP" if best == "LightGBM" else "Decomposition", style={"color": "#C9D1D9", "fontSize": "13px"}),
    ])

    return title, subtitle, cards, fig, img_src, interp_title, table_data, sidebar_info


if __name__ == "__main__":
    app.run(debug=True, port=8050)