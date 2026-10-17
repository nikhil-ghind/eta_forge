"""Feature engineering for the residual, prep-time and dwell models.

``spec``             the frozen name -> index contract and its version hash
``builder``          the one definition of every residual-model feature, used by both
                     training and serving
``calendar``         cyclical hour, weekday, rush hour, holidays (in local time)
``h3_cells``         origin/destination cells and the categorical vocabulary
``target_encoding``  smoothed, out-of-fold historical residual per cell
``weather``          nearest-station as-of join with a staleness bound
"""

from .spec import FeatureSpec, residual_spec, route_feature_names
from .builder import FeatureBuilder, FeatureRow, feature_hash
from .calendar import HolidayCalendar, calendar_features
from .h3_cells import CellVocabulary, cell_for, cell_scheme
from .target_encoding import TargetEncoder
from .weather import WeatherStation, WeatherTable, weather_features

__all__ = [
    "FeatureSpec",
    "residual_spec",
    "route_feature_names",
    "FeatureBuilder",
    "FeatureRow",
    "feature_hash",
    "HolidayCalendar",
    "calendar_features",
    "CellVocabulary",
    "cell_for",
    "cell_scheme",
    "TargetEncoder",
    "WeatherStation",
    "WeatherTable",
    "weather_features",
]
