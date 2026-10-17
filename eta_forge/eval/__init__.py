"""Offline evaluation: is the ETA accurate, is the p90 a p90, and where is it wrong.

``metrics``      MAE / floored MAPE / RMSE / bias / pinball / coverage / sharpness
``calibration``  observed vs nominal coverage across the implied quantiles
``slices``       every metric by hour, weekday, distance, cell and merchant
``harness``      routing-only vs routing + residual, on held-out trips
``report``       Markdown + CSV + JSON from one result
"""

from .metrics import mae, mape, point_metrics, quantile_metrics, rmse
from .calibration import calibration_curve, implied_quantile, inverse_normal_cdf
from .slices import EvalRecord, SliceMetrics, slice_metrics
from .harness import EvalResult, evaluate, records_from_rows, records_from_service
from .report import render_markdown, write_report

__all__ = [
    "mae",
    "mape",
    "rmse",
    "point_metrics",
    "quantile_metrics",
    "calibration_curve",
    "implied_quantile",
    "inverse_normal_cdf",
    "EvalRecord",
    "SliceMetrics",
    "slice_metrics",
    "EvalResult",
    "evaluate",
    "records_from_rows",
    "records_from_service",
    "render_markdown",
    "write_report",
]
