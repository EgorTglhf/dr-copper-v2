import numpy as np
import pandas as pd
import plotly.graph_objects as go
from arch import arch_model
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


def fit_garch(returns, dist="normal"):
    # Estimate in percentage-return units for numerical stability.
    model = arch_model(
        100 * returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist=dist,
        rescale=False,
    )
    return model.fit(disp="off")


def compare_garch_models(returns):
    fits = {
        name: fit_garch(returns, dist=distribution)
        for name, distribution in {"Gaussian": "normal", "Student-t": "t"}.items()
    }
    comparison = pd.DataFrame(
        {
            name: {
                "nobs": fit.nobs,
                "converged": fit.convergence_flag == 0,
                "BIC": fit.bic,
                "omega": fit.params["omega"],
                "alpha": fit.params["alpha[1]"],
                "beta": fit.params["beta[1]"],
                "persistence": fit.params["alpha[1]"] + fit.params["beta[1]"],
                "nu": fit.params.get("nu", np.nan),
            }
            for name, fit in fits.items()
        }
    ).T.sort_values("BIC")
    comparison["delta_BIC"] = comparison["BIC"] - comparison["BIC"].min()
    return comparison, fits


def garch_residual_summary(fits, lags=(5, 10, 20)):
    results = {}
    for name, fit in fits.items():
        standardized = fit.std_resid.dropna()
        signed = acorr_ljungbox(standardized, lags=lags)
        squared = acorr_ljungbox(standardized**2, lags=lags)
        results[name] = pd.DataFrame(
            {
                "residual_Q": signed["lb_stat"],
                "residual_p_value": signed["lb_pvalue"],
                "squared_Q": squared["lb_stat"],
                "squared_p_value": squared["lb_pvalue"],
            }
        )
    return pd.concat(results, names=["model", "lag"])


def forecast_garch_models(returns, splits):
    forecasts, refits = [], []
    for split in splits.itertuples():
        train = returns.loc[split.train_start : split.train_end]
        block = returns.loc[split.test_start : split.test_end]
        if block.empty:
            continue
        fold = split.test_start.strftime("%Y-%m")
        for name, distribution in {"Gaussian": "normal", "Student-t": "t"}.items():
            fit = fit_garch(train, dist=distribution)
            if fit.convergence_flag != 0:
                raise ValueError(f"{name} GARCH did not converge for {fold}.")
            omega = fit.params["omega"]
            alpha = fit.params["alpha[1]"]
            beta = fit.params["beta[1]"]
            persistence = alpha + beta
            refits.append(
                {
                    "fold": fold,
                    "model": name,
                    "fit_start": train.index[0],
                    "fit_end": train.index[-1],
                    "nobs": fit.nobs,
                    **fit.params.to_dict(),
                    "persistence": persistence,
                }
            )
            # Variance for the first block return, using only the preceding fit.
            state = (
                omega
                + alpha * fit.resid.iloc[-1] ** 2
                + beta * fit.conditional_volatility.iloc[-1] ** 2
            )
            for date, observed in block.items():
                # Consume today's completed return before forecasting tomorrow.
                next_variance = omega + alpha * (100 * observed) ** 2 + beta * state
                horizon_variance = next_variance
                cumulative_variance = next_variance
                for _ in range(2, 6):
                    horizon_variance = omega + persistence * horizon_variance
                    cumulative_variance += horizon_variance
                # Convert squared percentage-return units to decimal-return units.
                variance_1d = next_variance / 100**2
                variance_5d = cumulative_variance / 100**2
                if (
                    not np.all(np.isfinite([variance_1d, variance_5d]))
                    or min(variance_1d, variance_5d) <= 0
                ):
                    raise ValueError(f"Invalid {name} variance at {date}.")
                forecasts.append(
                    {
                        "date": date,
                        "model": name,
                        "fold": fold,
                        "fit_end": train.index[-1],
                        "variance_1d": variance_1d,
                        "variance_5d": variance_5d,
                        "volatility_1d": np.sqrt(variance_1d),
                        "volatility_5d": np.sqrt(variance_5d),
                    }
                )
                state = next_variance
    return pd.DataFrame(forecasts), pd.DataFrame(refits)


def compare_variance_forecasts(forecasts, returns, rolling_volatility):
    daily_benchmark = rolling_volatility.pow(2) / 252
    squared_returns = returns.pow(2)
    scores = []
    for horizon in (1, 5):
        actual = sum(squared_returns.shift(-step) for step in range(1, horizon + 1))
        predictions = forecasts.pivot(
            index="date", columns="model", values=f"variance_{horizon}d"
        ).reindex(returns.index)
        predictions["21-session rolling"] = horizon * daily_benchmark

        # Score every model on the same origins with a complete future window.
        matched = predictions.join(actual.rename("actual")).dropna()
        errors = matched.drop(columns="actual").sub(matched["actual"], axis=0)
        scores.append(
            pd.DataFrame({"origins": len(matched), "MSE": errors.pow(2).mean()})
        )
    return pd.concat(scores, keys=[1, 5], names=["sessions", "model"])


def plot_garch_forecasts(forecasts, close):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_scatter(
        x=forecasts.index,
        y=close.reindex(forecasts.index),
        mode="lines",
        name="CPER Close",
        hovertemplate="$%{y:.2f}",
        row=1,
        col=1,
    )
    for column, label in (
        ("volatility_1d", "Next session"),
        ("volatility_5d", "Next five sessions"),
    ):
        fig.add_scatter(
            x=forecasts.index,
            y=forecasts[column],
            mode="lines",
            name=label,
            hovertemplate="%{y:.2%}",
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="Close (USD)", row=1, col=1)
    fig.update_yaxes(
        title_text="Volatility (not annualized)", tickformat=".1%", row=2, col=1
    )
    fig.update_xaxes(title_text="Forecast origin (CPER session)", row=2, col=1)
    fig.update_layout(
        title="CPER Close and Gaussian GARCH volatility forecasts",
        height=630,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        margin=dict(t=100),
    )
    return fig


def plot_garch_residuals(fits, nlags=20):
    titles = [
        f"{name}: {kind}"
        for name in fits
        for kind in ("standardized residual ACF", "squared residual ACF", "Q-Q")
    ]
    fig = make_subplots(rows=len(fits), cols=3, subplot_titles=titles)
    for row, fit in enumerate(fits.values(), start=1):
        residuals = fit.std_resid.dropna()
        for col, values in enumerate((residuals, residuals**2), start=1):
            fig.add_trace(
                go.Bar(x=list(range(1, nlags + 1)), y=acf(values, nlags=nlags)[1:]),
                row=row,
                col=col,
            )
            for bound in (-1.96 / np.sqrt(len(values)), 1.96 / np.sqrt(len(values))):
                fig.add_hline(y=bound, line_dash="dash", row=row, col=col)
            fig.update_xaxes(title_text="Lag", dtick=5, row=row, col=col)

        probabilities = (np.arange(len(residuals)) + 0.5) / len(residuals)
        nu = fit.params.get("nu")
        theoretical = (
            stats.norm.ppf(probabilities)
            if nu is None
            else stats.t.ppf(probabilities, df=nu) * np.sqrt((nu - 2) / nu)
        )
        fig.add_trace(
            go.Scatter(
                x=theoretical,
                y=np.sort(residuals),
                mode="markers",
                marker=dict(size=4),
            ),
            row=row,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=theoretical[[0, -1]],
                y=theoretical[[0, -1]],
                mode="lines",
                line=dict(dash="dash"),
            ),
            row=row,
            col=3,
        )
        fig.update_xaxes(title_text="Fitted-distribution quantile", row=row, col=3)
        fig.update_yaxes(title_text="Residual quantile", row=row, col=3)
    fig.update_annotations(font_size=12)
    fig.update_layout(
        title="Data1: GARCH standardized residuals",
        width=1200,
        height=320 * len(fits),
        showlegend=False,
    )
    return fig
