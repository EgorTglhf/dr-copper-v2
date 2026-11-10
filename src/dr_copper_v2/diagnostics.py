import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, pacf

MONTHLY_FEATURES_PREFIX = "cli_"


def plot_return_dependence(returns: pd.Series, nlags: int = 20) -> go.Figure:
    centered = returns - returns.mean()
    values = [
        acf(returns, nlags=nlags)[1:],
        pacf(returns, nlags=nlags, method="ywm")[1:],
        acf(centered**2, nlags=nlags)[1:],
    ]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "ACF: daily returns",
            "PACF: daily returns",
            "ACF: squared demeaned returns",
        ],
    )
    for col, correlations in enumerate(values, start=1):
        fig.add_trace(
            go.Bar(x=list(range(1, nlags + 1)), y=correlations), row=1, col=col
        )
        for bound in [-1.96 / np.sqrt(len(returns)), 1.96 / np.sqrt(len(returns))]:
            fig.add_hline(y=bound, line_dash="dash", row=1, col=col)
    fig.update_xaxes(title_text="Lag (daily observations)", dtick=5)
    fig.update_annotations(font_size=12)
    fig.update_layout(
        title="Return dependence", width=1150, height=430, showlegend=False
    )
    return fig


def return_dependence_summary(returns: pd.Series, lags=(5, 10, 20)) -> pd.DataFrame:
    centered = returns - returns.mean()
    results = pd.concat(
        {
            "daily_return": acorr_ljungbox(returns, lags=lags),
            "squared_demeaned_return": acorr_ljungbox(centered**2, lags=lags),
        },
        names=["series", "lag"],
    )
    return results.rename(columns={"lb_stat": "Q", "lb_pvalue": "p_value"})


def mean_residual_summary(fits: dict, lags=(5, 10, 20)) -> pd.DataFrame:
    results = {}
    for name, fit in fits.items():
        ar_order = len(fit.model.ar_lags or [])
        residuals = acorr_ljungbox(fit.resid, lags=lags, model_df=ar_order)
        squares = acorr_ljungbox(fit.resid**2, lags=lags)
        results[name] = pd.DataFrame(
            {
                "residual_Q": residuals["lb_stat"],
                "residual_p_value": residuals["lb_pvalue"],
                "squared_Q": squares["lb_stat"],
                "squared_p_value": squares["lb_pvalue"],
            }
        )
    return pd.concat(results, names=["model", "lag"])


def plot_mean_residuals(fits: dict, nlags: int = 20) -> go.Figure:
    titles = [
        f"{name}: {kind}"
        for name in fits
        for kind in ("residual ACF", "squared residual ACF")
    ]
    fig = make_subplots(rows=len(fits), cols=2, subplot_titles=titles)
    for row, fit in enumerate(fits.values(), start=1):
        for col, values in enumerate((fit.resid, fit.resid**2), start=1):
            fig.add_trace(
                go.Bar(x=list(range(1, nlags + 1)), y=acf(values, nlags=nlags)[1:]),
                row=row,
                col=col,
            )
            for bound in (-1.96 / np.sqrt(len(values)), 1.96 / np.sqrt(len(values))):
                fig.add_hline(y=bound, line_dash="dash", row=row, col=col)
    fig.update_xaxes(title_text="Lag (daily observations)", dtick=5)
    fig.update_annotations(font_size=12)
    fig.update_layout(
        title="Mean-model residual dependence",
        width=1100,
        height=260 * len(fits),
        showlegend=False,
    )
    return fig


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
