import pandas as pd


def direction_feature_groups(features, garch, scores, regimes):
    index = features.index
    volatility = garch.loc[index, ["volatility_1d", "volatility_5d"]]
    scores = scores.loc[index]
    regime = pd.Categorical(regimes.loc[index, "regime"], categories=[0, 1])
    encoded = pd.get_dummies(regime, prefix="regime", dtype=int)
    encoded.index = index
    derived = scores.join(encoded)
    market_columns = features.columns[
        ~features.columns.str.startswith(("real_yield_", "cli_"))
    ]
    original = features.join(volatility)
    return {
        "market_garch": features[market_columns].join(volatility),
        "market_macro_garch": original,
        "market_macro_garch_pc_regimes": original.join(derived),
        "garch_pc_regimes": volatility.join(derived),
    }
