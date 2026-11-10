import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


def compare_mean_models(returns: pd.Series, max_lag: int = 20):
    specifications = [("Zero", 0, "n"), ("Constant", 0, "c")]
    specifications += [(f"AR({p})", p, "c") for p in range(1, max_lag + 1)]
    fits, rows = {}, []
    for name, order, trend in specifications:
        fit = AutoReg(
            returns.to_numpy(), lags=order, trend=trend, hold_back=max_lag
        ).fit()
        fits[name] = fit
        rows.append(
            {
                "model": name,
                "ar_order": order,
                "nobs": int(fit.nobs),
                "parameters": int(fit.df_model + 1),
                "BIC": fit.bic,
            }
        )
    comparison = pd.DataFrame(rows).set_index("model").sort_values("BIC")
    comparison["delta_BIC"] = comparison["BIC"] - comparison["BIC"].min()
    return comparison, fits


def forecast_mean_models(
    returns: pd.Series,
    splits: pd.DataFrame,
    horizons=(1, 5),
    hold_back: int = 1,
) -> pd.DataFrame:
    rows = []
    specifications = [("Zero", 0, "n"), ("Constant", 0, "c"), ("AR(1)", 1, "c")]
    for split in splits.itertuples():
        train = returns.loc[split.train_start : split.train_end]
        block = returns.loc[split.test_start : split.test_end]
        for name, order, trend in specifications:
            fit = AutoReg(
                train.to_numpy(), lags=order, trend=trend, hold_back=hold_back
            ).fit()
            intercept = 0.0 if name == "Zero" else fit.params[0]
            phi = fit.params[1] if order else 0.0
            for date, observed in block.items():
                state, cumulative = observed, 0.0
                for step in range(1, max(horizons) + 1):
                    state = intercept + phi * state
                    cumulative += state
                    if step in horizons:
                        rows.append(
                            {
                                "date": date,
                                "model": name,
                                "horizon": step,
                                "forecast": cumulative,
                                "fit_end": split.train_end,
                                "fold": split.test_start.strftime("%Y-%m"),
                            }
                        )
    return pd.DataFrame(rows)
