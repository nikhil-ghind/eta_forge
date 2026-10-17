"""Smoothed target encoding of high-cardinality keys (origin / destination cells).

The encoded value of a cell is its historical mean residual, shrunk toward the global
mean by the same credibility rule the speed profiles use:

    enc(cell) = (sum_residual + m * global_mean) / (n + m)

so a cell with 4 trips says almost nothing and a cell with 4,000 says a lot.

Target encoding is the single easiest place in this project to leak the answer. Two
rules are enforced here:

1. The encoder used for validation and test rows is fitted **on training rows only**.
2. Training rows themselves are encoded **out of fold** — never with statistics that
   include their own target. The default ``past`` mode goes further and encodes each
   time block using only *earlier* blocks, which is exactly the information a model
   in production would have had at that moment.

Break either rule and offline p50 MAE improves by an amount that evaporates the day
the model ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple


@dataclass
class TargetEncoder:
    smoothing: float = 30.0
    global_mean: float = 0.0
    stats: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.smoothing <= 0:
            raise ValueError("target encoding smoothing must be > 0")

    # -- fitting -----------------------------------------------------------------

    def fit(self, keys: Sequence[Hashable], targets: Sequence[float]) -> "TargetEncoder":
        if len(keys) != len(targets):
            raise ValueError("keys and targets must have the same length")
        self.stats = {}
        total = 0.0
        for key, y in zip(keys, targets):
            n, s = self.stats.get(str(key), (0, 0.0))
            self.stats[str(key)] = (n + 1, s + float(y))
            total += float(y)
        self.global_mean = total / len(targets) if targets else 0.0
        return self

    def encode_with(
        self, key: Hashable, stats: Mapping[str, Tuple[int, float]], prior: float
    ) -> float:
        n, s = stats.get(str(key), (0, 0.0))
        return (s + self.smoothing * prior) / (n + self.smoothing)

    def transform_one(self, key: Optional[Hashable]) -> float:
        if key is None:
            return self.global_mean
        return self.encode_with(key, self.stats, self.global_mean)

    def transform(self, keys: Sequence[Optional[Hashable]]) -> List[float]:
        return [self.transform_one(k) for k in keys]

    def count(self, key: Hashable) -> int:
        return self.stats.get(str(key), (0, 0.0))[0]

    def fit_transform_oof(
        self,
        keys: Sequence[Hashable],
        targets: Sequence[float],
        folds: Sequence[Sequence[int]],
        *,
        mode: str = "past",
    ) -> List[float]:
        """Fit on all rows, and return out-of-fold encodings for those same rows.

        ``folds`` are lists of row positions, in time order. In ``past`` mode fold
        ``f`` is encoded from folds ``0..f-1`` only (fold 0 gets the prior from fold 0's
        own *global* mean, with no per-key information). In ``kfold`` mode it is
        encoded from every other fold.
        """
        if mode not in ("past", "kfold"):
            raise ValueError("mode must be 'past' or 'kfold'")
        self.fit(keys, targets)
        encoded: List[Optional[float]] = [None] * len(keys)

        fold_stats: List[Dict[str, Tuple[int, float]]] = []
        for fold in folds:
            stats: Dict[str, Tuple[int, float]] = {}
            for i in fold:
                n, s = stats.get(str(keys[i]), (0, 0.0))
                stats[str(keys[i])] = (n + 1, s + float(targets[i]))
            fold_stats.append(stats)

        for f, fold in enumerate(folds):
            sources = range(f) if mode == "past" else [g for g in range(len(folds)) if g != f]
            merged: Dict[str, Tuple[int, float]] = {}
            for g in sources:
                for key, (n, s) in fold_stats[g].items():
                    mn, ms = merged.get(key, (0, 0.0))
                    merged[key] = (mn + n, ms + s)
            count = sum(n for n, _ in merged.values())
            if count:
                prior = sum(s for _, s in merged.values()) / count
            else:
                prior = (
                    sum(float(targets[i]) for i in fold) / len(fold) if fold else self.global_mean
                )
            for i in fold:
                encoded[i] = self.encode_with(keys[i], merged, prior)

        return [self.global_mean if v is None else v for v in encoded]

    # -- persistence -------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smoothing": self.smoothing,
            "global_mean": self.global_mean,
            "stats": {k: [n, s] for k, (n, s) in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetEncoder":
        return cls(
            smoothing=float(data.get("smoothing", 30.0)),
            global_mean=float(data.get("global_mean", 0.0)),
            stats={
                str(k): (int(v[0]), float(v[1]))
                for k, v in dict(data.get("stats", {})).items()
            },
        )


__all__ = ["TargetEncoder"]
