"""Stage one: the physical routing engine.

A baseline duration comes out of a real road graph, not out of a distance heuristic.
The pieces:

``geometry``  great-circle distance, bearings, and the angle arithmetic everything else
              depends on
``turns``     bearing deltas classified into manoeuvres, and the time they cost
``costs``     composition of the ``pgr_dijkstra`` cost expression, including the mined
              speed-profile lookup that makes 08:00 Monday a different graph from
              03:00 Sunday
``router``    KNN snapping, the Dijkstra call, and assembly into a ``BaselineRoute``
``summary``   the structural descriptors of a route the learner consumes as features
``synthetic`` an in-memory grid graph implementing the same interface, so everything
              downstream runs with no PostGIS
"""

from .geometry import (
    bearing_deg,
    bearing_delta_deg,
    haversine_m,
    normalize_bearing,
    polyline_length_m,
)
from .turns import classify_turn, route_turn_profile, turn_penalty_s
from .costs import CostExpression, build_cost_sql, time_bucket
from .summary import RouteSummary, summarize_route
from .router import Router, RouterProtocol, SnapResult
from .synthetic import SyntheticGraph

__all__ = [
    "bearing_deg",
    "bearing_delta_deg",
    "haversine_m",
    "normalize_bearing",
    "polyline_length_m",
    "classify_turn",
    "route_turn_profile",
    "turn_penalty_s",
    "CostExpression",
    "build_cost_sql",
    "time_bucket",
    "RouteSummary",
    "summarize_route",
    "Router",
    "RouterProtocol",
    "SnapResult",
    "SyntheticGraph",
]
