"""Time-of-day x day-of-week speed profiles per road segment.

This is the feedback loop that makes stage one worth having. Historical trips are
map-matched onto segments; each traversal contributes an observed speed to a
``(segment, dow, tod)`` cell; the cells are smoothed against a hierarchy of priors; and
the smoothed table becomes the edge cost that pgRouting routes on. The baseline then
knows that this stretch of Canal Street moves at 9 km/h at 18:00 on a Friday.

Three things are easy to get wrong here, and each has its own module:

``buckets``    which cell a timestamp belongs to, in local time, consistently between
               the miner and the router
``mining``     the *harmonic* mean — averaging speeds over a fixed distance with an
               arithmetic mean systematically overestimates
``shrinkage``  a cell with three observations must not override the class prior
"""

from .buckets import BucketSpec, bucket_of, bucket_label, iter_cells
from .mining import CellStats, SegmentProfile, mine_profiles, harmonic_mean_speed
from .shrinkage import PriorHierarchy, shrink_cell, shrink_profiles
from .store import ProfileBuild, ProfileStore, build_profiles

__all__ = [
    "BucketSpec",
    "bucket_of",
    "bucket_label",
    "iter_cells",
    "CellStats",
    "SegmentProfile",
    "mine_profiles",
    "harmonic_mean_speed",
    "PriorHierarchy",
    "shrink_cell",
    "shrink_profiles",
    "ProfileStore",
    "ProfileBuild",
    "build_profiles",
]
