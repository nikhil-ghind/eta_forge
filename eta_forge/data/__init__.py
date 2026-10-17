"""Trip ingestion: the TLC loader, its quality filter, and the synthetic source.

``tlc``        streaming reader over every TLC schema the commission has published,
               normalised onto one record with timestamps localised to New York
``quality``    plausibility rules with a per-reason rejection tally, so data loss is
               a number in the ingest ledger and not a surprise in the model
``synthetic``  seeded trips, GPS traces and ground-truth traversals over the
               synthetic grid, so the pipeline runs end to end with no downloads
"""

from .quality import QualityRules, RejectionTally, check_trip
from .tlc import LoadResult, NormalizedTrip, TlcLoader, ZoneCentroids
from .synthetic import SyntheticConfig, SyntheticSource, SyntheticTrip

__all__ = [
    "QualityRules",
    "RejectionTally",
    "check_trip",
    "LoadResult",
    "NormalizedTrip",
    "TlcLoader",
    "ZoneCentroids",
    "SyntheticConfig",
    "SyntheticSource",
    "SyntheticTrip",
]
