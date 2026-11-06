import pandas as pd


def walk_forward_split(
    data: pd.DataFrame,
    test_start: str = "2017-01-01",
    test_end: str = "2023-01-01",
):
    stop = pd.Timestamp(test_end)
    for start in pd.date_range(test_start, stop, freq="MS", inclusive="left"):
        end = min(start + pd.offsets.MonthBegin(1), stop)
        train = data.loc[data.index < start]
        test = data.loc[(data.index >= start) & (data.index < end)]
        yield train, test
