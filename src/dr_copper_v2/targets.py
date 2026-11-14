import pandas as pd


def direction_target(close: pd.Series, horizon: int = 5) -> pd.DataFrame:
    forward_return = close.shift(-horizon).div(close).sub(1)
    label_end = close.index.to_series().shift(-horizon)
    direction = forward_return.gt(0).astype("Int8").where(forward_return.notna())
    return pd.DataFrame(
        {
            "label_end": label_end,
            "forward_return": forward_return,
            "direction": direction,
        },
        index=close.index,
    )
