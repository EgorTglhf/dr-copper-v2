import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_pca(features: pd.DataFrame, n_components: int | None = None):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, svd_solver="full", whiten=False)),
        ]
    )
    pipe.fit(features)
    pca = pipe.named_steps["pca"]
    labels = [f"PC{i + 1}" for i in range(pca.n_components_)]
    loadings = pd.DataFrame(pca.components_.T, index=features.columns, columns=labels)
    scores = transform_pca(features, pipe)
    evr = pd.Series(pca.explained_variance_ratio_, index=labels, name="variance_ratio")
    return pipe, loadings, scores, evr


def transform_pca(features: pd.DataFrame, pipe: Pipeline) -> pd.DataFrame:
    values = pipe.transform(features[list(pipe.feature_names_in_)])
    labels = [f"PC{i + 1}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=features.index, columns=labels)


def compare_loadings(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    left = reference.to_numpy()
    right = current.loc[reference.index].to_numpy()
    left = left / np.linalg.norm(left, axis=0)
    right = right / np.linalg.norm(right, axis=0)
    similarity = np.abs(left.T @ right).clip(0, 1)
    rows, cols = linear_sum_assignment(-similarity)
    return pd.DataFrame(
        {
            "component": reference.columns[rows],
            "matched_component": current.columns[cols],
            "similarity": similarity[rows, cols],
        }
    ).set_index("component")
