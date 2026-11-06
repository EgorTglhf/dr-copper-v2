import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller

MONTHLY_FEATURES_PREFIX = "cli_"


def _diagnostic_series(series: pd.Series) -> pd.Series:
    # Daily carried CLI levels do not constitute fresh daily observations.
    if series.name.startswith(MONTHLY_FEATURES_PREFIX):
        return series.resample("ME").last().dropna()
    return series.dropna()


def adf_summary(
    features: pd.DataFrame, regression: str = "c", alpha: float = 0.05
) -> pd.DataFrame:
    rows = []
    for feature in features:
        series = _diagnostic_series(features[feature])
        result = adfuller(
            series, regression=regression, autolag="AIC", result_object=True
        )
        rows.append(
            {
                "feature": feature,
                "frequency": (
                    "month_end"
                    if feature.startswith(MONTHLY_FEATURES_PREFIX)
                    else "session"
                ),
                "n_values": len(series),
                "regression": regression,
                "adf_statistic": result.statistic,
                "p_value": result.pvalue,
                "lags_used": result.lags,
                "nobs": result.nobs,
                "critical_5pct": result.critical_values["5%"],
                "reject_unit_root": result.pvalue < alpha,
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def plot_feature_diagnostics(
    features: pd.DataFrame, daily_window: int = 63, monthly_window: int = 12
) -> go.Figure:
    titles = []
    for feature in features:
        window = (
            monthly_window
            if feature.startswith(MONTHLY_FEATURES_PREFIX)
            else daily_window
        )
        titles.extend(
            [feature, f"Trailing mean ({window} rows)", f"Trailing std ({window} rows)"]
        )

    fig = make_subplots(
        rows=len(features.columns),
        cols=3,
        subplot_titles=titles,
        vertical_spacing=min(0.08, 0.6 / len(features.columns)),
    )
    for row, feature in enumerate(features, start=1):
        series = _diagnostic_series(features[feature])
        window = (
            monthly_window
            if feature.startswith(MONTHLY_FEATURES_PREFIX)
            else daily_window
        )
        for col, values in enumerate(
            [series, series.rolling(window).mean(), series.rolling(window).std()],
            start=1,
        ):
            fig.add_trace(
                go.Scatter(x=values.index, y=values, mode="lines", name=feature),
                row=row,
                col=col,
            )
    fig.update_layout(
        title="Feature diagnostics",
        width=1100,
        height=300 * len(features.columns),
        font=dict(size=13),
        showlegend=False,
    )
    fig.update_annotations(font_size=12)
    fig.update_xaxes(dtick="M12", tickformat="%Y")
    return fig
