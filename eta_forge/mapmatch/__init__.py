"""HMM map matching: GPS traces -> road segments -> per-segment traversal times.

``candidates``   the network interface the matcher needs (nearby segments, driven
                 distance between positions), over the synthetic grid or PostGIS
``hmm``          Gaussian emissions, Newson-Krumm exponential transitions, log-space
                 Viterbi with lattice breaks
``attribution``  elapsed time spread over the decoded path, kept only for segments
                 seen from entry to exit
"""

from .candidates import (
    Candidate,
    CandidateSource,
    GraphCandidateSource,
    PathBetween,
    PathPiece,
    PostgisCandidateSource,
)
from .hmm import HmmMapMatcher, HmmParams, MatchResult, MatchedPoint, viterbi
from .attribution import AttributedTraversal, attribute_traversals, trip_segment_rows

__all__ = [
    "Candidate",
    "CandidateSource",
    "GraphCandidateSource",
    "PathBetween",
    "PathPiece",
    "PostgisCandidateSource",
    "HmmMapMatcher",
    "HmmParams",
    "MatchResult",
    "MatchedPoint",
    "viterbi",
    "AttributedTraversal",
    "attribute_traversals",
    "trip_segment_rows",
]
