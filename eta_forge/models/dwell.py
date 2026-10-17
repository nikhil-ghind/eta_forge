"""Handoff dwell: the last-metre term of ``prep + travel + handoff``.

Arriving at the kerb is not arriving at the customer. Finding parking, a doorman, a
lobby, a lift to the 31st floor: that is a property of the *built environment* at the
destination, not of the courier or the hour, so it is modelled per destination H3 cell.

Each cell's empirical p50/p90 dwell is shrunk toward the city-wide one. The ratio of a
cell's p90 to the city p90 is exported as a parking-difficulty proxy: Midtown towers
sit well above 1, brownstone Brooklyn below it.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..types import DwellObservation, QuantilePair
from .quantile import QuantilePrior, coverage, empirical_prior, shrink_prior

DWELL_FILE = "dwell.json"


class DwellEstimator:
    def __init__(
        self,
        global_prior: QuantilePrior,
        cells: Optional[Mapping[str, QuantilePrior]] = None,
        *,
        k: float = 40.0,
    ):
        self.global_prior = global_prior
        self.cells: Dict[str, QuantilePrior] = dict(cells or {})
        self.k = k

    @classmethod
    def default(cls, p50_s: float = 90.0, p90_s: float = 240.0) -> "DwellEstimator":
        return cls(QuantilePrior(p50_s, max(p50_s, p90_s), 0))

    @classmethod
    def fit(cls, observations: Sequence[DwellObservation], *, k: float = 40.0,
            fallback: Optional[QuantilePrior] = None) -> "DwellEstimator":
        values = [o.dwell_s for o in observations]
        global_prior = empirical_prior(values) or fallback or QuantilePrior(90.0, 240.0, 0)
        by_cell: Dict[str, List[float]] = defaultdict(list)
        for o in observations:
            by_cell[o.destination_cell].append(o.dwell_s)
        cells = {
            cell: shrink_prior(empirical_prior(v), global_prior, k) for cell, v in by_cell.items()
        }
        return cls(global_prior, cells, k=k)

    def prior(self, cell: Optional[str]) -> QuantilePrior:
        if cell is not None and cell in self.cells:
            return self.cells[cell]
        return self.global_prior

    def predict(self, cell: Optional[str]) -> QuantilePair:
        prior = self.prior(cell)
        return QuantilePair(prior.p50, prior.p90)

    def parking_difficulty(self, cell: Optional[str]) -> float:
        if self.global_prior.p90 <= 0:
            return 1.0
        return self.prior(cell).p90 / self.global_prior.p90

    def evaluate(self, observations: Sequence[DwellObservation]) -> Dict[str, float]:
        if not observations:
            return {}
        actual = [o.dwell_s for o in observations]
        p90 = [self.predict(o.destination_cell).p90 for o in observations]
        p50 = [self.predict(o.destination_cell).p50 for o in observations]
        return {
            "rows": float(len(actual)),
            "mae_p50": sum(abs(a - p) for a, p in zip(actual, p50)) / len(actual),
            "coverage_p90": coverage(actual, p90),
        }

    def save(self, directory: str) -> Dict[str, Any]:
        os.makedirs(directory, exist_ok=True)
        payload = {
            "kind": "dwell",
            "k": self.k,
            "global": self.global_prior.to_list(),
            "cells": {c: p.to_list() for c, p in self.cells.items()},
        }
        with open(os.path.join(directory, DWELL_FILE), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        return {"kind": "dwell", "cells": len(self.cells), "k": self.k}

    @classmethod
    def load(cls, directory: str) -> "DwellEstimator":
        with open(os.path.join(directory, DWELL_FILE), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            QuantilePrior.from_list(data["global"]),
            {c: QuantilePrior.from_list(v) for c, v in data.get("cells", {}).items()},
            k=float(data.get("k", 40.0)),
        )


__all__ = ["DwellEstimator", "DWELL_FILE"]
