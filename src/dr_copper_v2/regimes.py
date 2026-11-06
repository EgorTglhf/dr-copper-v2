import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def fit_regimes(scores: pd.DataFrame, n_regimes: int = 3, random_state: int = 42):
    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    labels = pd.Series(km.fit_predict(scores), index=scores.index, name="regime")
    summary = scores.groupby(labels).mean()
    summary["n_obs"] = labels.value_counts().reindex(summary.index)
    summary["share"] = summary.n_obs / len(scores)
    return labels, km, summary


def predict_regimes(scores: pd.DataFrame, km: KMeans) -> pd.Series:
    values = km.predict(scores[list(km.feature_names_in_)])
    return pd.Series(values, index=scores.index, name="regime")


def regime_diagnostics(scores: pd.DataFrame, cluster_counts=range(2, 7)):
    rows = []
    for k in cluster_counts:
        labels, km, summary = fit_regimes(scores, n_regimes=k)
        rows.append(
            {
                "k": k,
                "inertia_per_row": km.inertia_ / len(scores),
                "silhouette": silhouette_score(
                    scores, labels, sample_size=min(500, len(scores)), random_state=42
                ),
                "smallest_share": summary.share.min(),
            }
        )
    return pd.DataFrame(rows).set_index("k")
