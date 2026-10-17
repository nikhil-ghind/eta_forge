"""Spatial cells for origin/destination features.

H3 resolution 8 (~0.74 km^2 hexagons, ~0.5 km edge) is the unit the residual model
reasons about space in: small enough that Midtown and the Upper West Side are
different cells, large enough that a cell has hundreds of historical trips to
target-encode from.

``h3`` is imported lazily. Without it, cells fall back to a deterministic square grid
of comparable size. The fallback ids are prefixed ``g`` so they can never be confused
with real H3 indexes, and a model records which scheme it was trained with: the cell
scheme is part of the feature contract, exactly like a column order.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from ..types import Location

# Approximate H3 hexagon edge length in metres, by resolution.
H3_EDGE_M = {5: 9_854.0, 6: 3_725.0, 7: 1_406.0, 8: 531.0, 9: 201.0, 10: 76.0, 11: 29.0}

# The fallback grid is laid out in a fixed equirectangular frame. A fixed reference
# latitude (rather than each point's own) keeps cell boundaries straight lines, so two
# nearby points never land in different cells because of a cosine rounding.
_REFERENCE_LAT = 40.7
_M_PER_DEG_LAT = 111_320.0


def _h3_module() -> Any:
    try:
        import h3  # lazily: optional dependency

        return h3
    except ImportError:
        return None


def h3_available() -> bool:
    return _h3_module() is not None


def cell_scheme() -> str:
    return "h3" if h3_available() else "grid"


def fallback_cell(location: Location, resolution: int) -> str:
    """Square-grid stand-in for an H3 cell of the given resolution."""
    edge = H3_EDGE_M.get(resolution, 531.0 * (2.65 ** (8 - resolution)))
    # A hexagon of edge e has area ~2.6 e^2; a square of side 1.6 e matches it.
    side = 1.6 * edge
    m_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(_REFERENCE_LAT))
    i = math.floor(location.lat * _M_PER_DEG_LAT / side)
    j = math.floor(location.lon * m_per_deg_lon / side)
    return f"g{resolution}:{i}:{j}"


def cell_for(location: Location, resolution: int, *, scheme: Optional[str] = None) -> str:
    """The cell id of a point. ``scheme`` pins ``h3`` or ``grid`` explicitly."""
    use = scheme or cell_scheme()
    if use == "h3":
        h3 = _h3_module()
        if h3 is None:
            raise RuntimeError("cell scheme 'h3' requested but the h3 package is missing")
        if hasattr(h3, "latlng_to_cell"):  # h3 >= 4
            return str(h3.latlng_to_cell(location.lat, location.lon, resolution))
        return str(h3.geo_to_h3(location.lat, location.lon, resolution))  # pragma: no cover
    return fallback_cell(location, resolution)


@dataclass
class CellVocabulary:
    """Cell id -> small dense integer, for LightGBM categorical splits.

    Only cells seen at least ``min_count`` times in training get their own code; the
    rest share code 0 ("other"). Without that floor the categorical has tens of
    thousands of levels, most seen once, and LightGBM will happily split on them.
    """

    codes: Dict[str, int] = field(default_factory=dict)
    min_count: int = 20
    max_size: int = 2_000

    def fit(self, cells: Iterable[str]) -> "CellVocabulary":
        counts = Counter(cells)
        frequent = [
            cell for cell, n in counts.most_common(self.max_size) if n >= self.min_count
        ]
        # Codes are assigned in sorted order so two fits on the same data agree
        # regardless of Counter's tie-breaking.
        self.codes = {cell: index + 1 for index, cell in enumerate(sorted(frequent))}
        return self

    def encode(self, cell: Optional[str]) -> int:
        if cell is None:
            return 0
        return self.codes.get(cell, 0)

    def __len__(self) -> int:
        return len(self.codes)

    def to_dict(self) -> Dict[str, Any]:
        return {"codes": dict(self.codes), "min_count": self.min_count, "max_size": self.max_size}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CellVocabulary":
        return cls(
            codes={str(k): int(v) for k, v in dict(data.get("codes", {})).items()},
            min_count=int(data.get("min_count", 20)),
            max_size=int(data.get("max_size", 2_000)),
        )


__all__ = [
    "CellVocabulary",
    "cell_for",
    "cell_scheme",
    "fallback_cell",
    "h3_available",
    "H3_EDGE_M",
]
