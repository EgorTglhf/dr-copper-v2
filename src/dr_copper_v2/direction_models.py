import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def forecast_xgb(folds, feature_group, params):
    forecasts = []
    for month, fold in folds.items():
        model = XGBClassifier(**params)
        model.fit(
            fold["inputs"]["train"][feature_group],
            fold["train_labels"]["direction"].astype(int),
        )
        output = model.predict_proba(fold["inputs"]["forecast"][feature_group])[:, 1]
        block = fold["forecast_labels"][["label_end", "direction"]].copy()
        block["fold"] = month
        block["model_output"] = output
        block["prediction"] = (output > 0.5).astype("int8")
        forecasts.append(block)
    return pd.concat(forecasts).sort_index()


def forecast_direction_benchmarks(folds, feature_group, logistic_candidates):
    forecasts = []
    for month, fold in folds.items():
        train = fold["inputs"]["train"][feature_group]
        forecast = fold["inputs"]["forecast"][feature_group]
        direction = fold["train_labels"]["direction"].astype(int)
        block = fold["forecast_labels"][["label_end", "direction"]].copy()
        block["fold"] = month
        block["always_up"] = 1
        for name, params in logistic_candidates.items():
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(l1_ratio=0.0, solver="lbfgs", **params),
            )
            model.fit(train, direction)
            output = model.predict_proba(forecast)[:, 1]
            block[name] = (output > 0.5).astype("int8")
        forecasts.append(block)
    return pd.concat(forecasts).sort_index()
