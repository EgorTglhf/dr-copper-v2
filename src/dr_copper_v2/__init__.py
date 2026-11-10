"""Dr. Copper v2 quantitative research pipeline."""

from dr_copper_v2.data_loader import fetch_raw
from dr_copper_v2.diagnostics import (
    adf_summary,
    mean_residual_summary,
    plot_feature_diagnostics,
    plot_mean_residuals,
    plot_return_dependence,
    return_dependence_summary,
)
from dr_copper_v2.feature_builder import add_macro_changes, fetch_features
from dr_copper_v2.mean_models import compare_mean_models, forecast_mean_models
from dr_copper_v2.pca import compare_loadings, fit_pca, transform_pca
from dr_copper_v2.regimes import fit_regimes, predict_regimes, regime_diagnostics
from dr_copper_v2.splitting import walk_forward_split

__all__ = [
    "fetch_raw",
    "fetch_features",
    "add_macro_changes",
    "adf_summary",
    "plot_feature_diagnostics",
    "plot_return_dependence",
    "return_dependence_summary",
    "mean_residual_summary",
    "plot_mean_residuals",
    "compare_mean_models",
    "forecast_mean_models",
    "walk_forward_split",
    "fit_pca",
    "transform_pca",
    "compare_loadings",
    "fit_regimes",
    "predict_regimes",
    "regime_diagnostics",
]
