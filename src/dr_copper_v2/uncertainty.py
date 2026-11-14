import pandas as pd
from arch.bootstrap import CircularBlockBootstrap


def accuracy_difference_intervals(
    directions, actual, model="XGBoost", block_sizes=(5, 21, 63), reps=5000, seed=42
):
    correct = directions.eq(actual, axis=0).astype(float)
    differences = correct.drop(columns=model).rsub(correct[model], axis=0)
    results = {}
    for block_size in block_sizes:
        bootstrap = CircularBlockBootstrap(block_size, differences, seed=seed)
        interval = bootstrap.conf_int(
            lambda sample: sample.mean(axis=0).to_numpy(),
            reps=reps,
            method="percentile",
            size=0.95,
        )
        results[block_size] = pd.DataFrame(
            {
                "accuracy_gain_pp": 100 * differences.mean(),
                "lower_95_pp": 100 * interval[0],
                "upper_95_pp": 100 * interval[1],
            }
        )
    return pd.concat(results, names=["block_rows", "benchmark"])
