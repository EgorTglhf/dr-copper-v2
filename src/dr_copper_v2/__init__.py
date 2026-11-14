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
from dr_copper_v2.direction_features import direction_feature_groups
from dr_copper_v2.direction_models import forecast_direction_benchmarks, forecast_xgb
from dr_copper_v2.feature_builder import add_macro_changes, fetch_features
from dr_copper_v2.mean_models import compare_mean_models, forecast_mean_models
from dr_copper_v2.pca import compare_loadings, fit_pca, transform_pca
from dr_copper_v2.regimes import fit_regimes, predict_regimes, regime_diagnostics
from dr_copper_v2.splitting import walk_forward_split
from dr_copper_v2.targets import direction_target
from dr_copper_v2.uncertainty import accuracy_difference_intervals
from dr_copper_v2.volatility import (
    compare_garch_models,
    compare_variance_forecasts,
    fit_garch,
    forecast_garch_models,
    garch_residual_summary,
    plot_garch_forecasts,
    plot_garch_residuals,
)

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
    "direction_target",
    "direction_feature_groups",
    "forecast_xgb",
    "forecast_direction_benchmarks",
    "accuracy_difference_intervals",
    "fit_pca",
    "transform_pca",
    "compare_loadings",
    "fit_regimes",
    "predict_regimes",
    "regime_diagnostics",
    "fit_garch",
    "forecast_garch_models",
    "compare_garch_models",
    "compare_variance_forecasts",
    "garch_residual_summary",
    "plot_garch_forecasts",
    "plot_garch_residuals",
]
