"""Stage two: the learned corrections, and the registry that versions them.

``quantile``   pinball loss, coverage, and the LightGBM / pure-Python booster backends
``splits``     time-ordered train / validation / test with an optional embargo
``dataset``    trips re-routed through the serving router into training rows
``residual``   p50/p90 corrections to the baseline, with monotonicity repair
``prep_time``  per-merchant prep quantiles, shrunk to cuisine, plus load corrections
``dwell``      handoff dwell per destination cell
``registry``   content-addressed versions, atomic aliases, rollback
"""

from .quantile import (
    StumpQuantileBooster,
    coverage,
    make_quantile_model,
    mean_pinball_loss,
    pinball_loss,
)
from .splits import TimeSplit, time_split
from .dataset import TrainingRow, build_rows
from .residual import ResidualModel, ResidualTrainer, TrainingReport
from .prep_time import PrepTimeModel
from .dwell import DwellEstimator
from .registry import ModelRegistry, ModelVersion, RegistryError

__all__ = [
    "StumpQuantileBooster",
    "coverage",
    "make_quantile_model",
    "mean_pinball_loss",
    "pinball_loss",
    "TimeSplit",
    "time_split",
    "TrainingRow",
    "build_rows",
    "ResidualModel",
    "ResidualTrainer",
    "TrainingReport",
    "PrepTimeModel",
    "DwellEstimator",
    "ModelRegistry",
    "ModelVersion",
    "RegistryError",
]
