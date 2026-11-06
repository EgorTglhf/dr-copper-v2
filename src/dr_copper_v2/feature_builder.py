import numpy as np
import pandas as pd

PRICE_COLS = ["copper", "gold", "crude", "sp500", "dxy", "USDCNY"]
REAL_YIELD_COL = "real_yield_10y_lag"
MACRO_COLS = ["cli_usa_latest_available", "cli_china_latest_available"]


def fetch_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    log_prices = np.log(raw[PRICE_COLS])
    for col in PRICE_COLS:
        out[f"{col}_ret1d"] = log_prices[col].diff(1)
        out[f"{col}_ret5d"] = log_prices[col].diff(5)

    out["copper_vol21d"] = log_prices["copper"].diff(1).rolling(21).std() * np.sqrt(252)

    macro_cols = [col for col in raw.columns if col.startswith(REAL_YIELD_COL)]
    macro_cols += MACRO_COLS
    out[macro_cols] = raw[macro_cols]

    return out


def add_macro_changes(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    yield_cols = [
        col
        for col in features
        if col.startswith(REAL_YIELD_COL) and not col.endswith("_change1d")
    ]
    for col in yield_cols:
        out[f"{col}_change1d"] = features[col].diff(1)

    monthly_cli = features[MACRO_COLS].resample("MS").first(skipna=False)
    monthly_changes = monthly_cli.diff()
    for col in MACRO_COLS:
        out[f"{col}_change1m"] = monthly_changes[col].reindex(
            features.index, method="ffill"
        )

    return out
