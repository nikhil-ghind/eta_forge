"""ETA Forge — two-stage ETA prediction for a ride-hail / delivery marketplace.

Stage one is physical: a pgRouting shortest path over an OSM-derived road graph whose
edge costs come from mined time-of-day speed profiles, producing a baseline duration.
Stage two is learned: a LightGBM quantile model that predicts the *residual* between
that baseline and observed reality, at p50 and p90.

Nothing in this package imports psycopg, lightgbm, pandas, numpy, h3 or fastapi at
module import time; every heavy dependency is imported inside the function that needs
it. That keeps the pure-Python core (geometry, bucketing, shrinkage, quantile math,
calibration, the registry) importable and unit-testable on a bare interpreter.
"""

__version__ = "0.5.0"

__all__ = ["__version__"]
