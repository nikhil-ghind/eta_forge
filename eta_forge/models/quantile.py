"""Quantile regression: the loss, the coverage arithmetic, and two model backends.

The residual, prep and dwell models all predict *quantiles*, not means. The loss is the
pinball (check) loss,

    L_alpha(y, q) = alpha * (y - q)        if y >= q
                    (1 - alpha) * (q - y)  otherwise

whose minimiser over a sample is exactly its alpha-quantile. At alpha = 0.9 an
under-prediction costs nine times what an over-prediction does, which is the
asymmetry an ETA *promise* needs.

Two backends implement the same small interface:

* :class:`LightGbmQuantile` — ``LGBMRegressor(objective='quantile')``, the production
  path. ``lightgbm`` and ``numpy`` are imported only when one is constructed.
* :class:`StumpQuantileBooster` — gradient boosting of depth-one trees under pinball
  loss, in pure Python. It is far weaker than LightGBM and exists so that the whole
  pipeline (train -> register -> serve -> evaluate) can run, and be tested, on a bare
  interpreter. Leaf values are refitted as the alpha-quantile of the residuals in
  each leaf, which is what LightGBM's quantile objective does too, and is why the
  booster converges on the quantile rather than drifting toward the median.
"""

from __future__ import annotations

import bisect
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..config import ModelConfig
from ..profiles.mining import percentile


def pinball_loss(y: float, prediction: float, alpha: float) -> float:
    diff = y - prediction
    return alpha * diff if diff >= 0 else (alpha - 1.0) * diff


def mean_pinball_loss(ys: Sequence[float], predictions: Sequence[float], alpha: float) -> float:
    if len(ys) != len(predictions):
        raise ValueError("ys and predictions must have the same length")
    if not ys:
        return 0.0
    return sum(pinball_loss(y, p, alpha) for y, p in zip(ys, predictions)) / len(ys)


def coverage(ys: Sequence[float], upper: Sequence[float]) -> float:
    """Fraction of actuals at or below the predicted quantile."""
    if len(ys) != len(upper):
        raise ValueError("ys and upper must have the same length")
    if not ys:
        return 0.0
    return sum(1 for y, u in zip(ys, upper) if y <= u) / len(ys)


def crossing_rate(lower: Sequence[float], upper: Sequence[float]) -> float:
    """How often independently fitted quantiles cross (``upper < lower``)."""
    if not lower:
        return 0.0
    return sum(1 for lo, hi in zip(lower, upper) if hi < lo) / len(lower)


def enforce_monotone(lower: Sequence[float], upper: Sequence[float]) -> List[Tuple[float, float]]:
    """Repair crossed pairs by lifting the upper quantile to the lower one."""
    return [(lo, max(lo, hi)) for lo, hi in zip(lower, upper)]


class QuantileModel(Protocol):
    alpha: float
    kind: str

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        *,
        X_val: Optional[Sequence[Sequence[float]]] = None,
        y_val: Optional[Sequence[float]] = None,
    ) -> "QuantileModel": ...

    def predict(self, rows: Sequence[Sequence[float]]) -> List[float]: ...

    def save(self, directory: str, stem: str) -> Dict[str, Any]: ...


# --------------------------------------------------------------------------------------
# pure-Python backend
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stump:
    feature: int
    threshold: float
    left: float
    right: float

    def value(self, row: Sequence[float]) -> float:
        x = row[self.feature]
        # NaN goes left, the same side it was binned to during training.
        return self.left if (x != x or x <= self.threshold) else self.right


@dataclass
class StumpQuantileBooster:
    alpha: float
    n_rounds: int = 150
    learning_rate: float = 0.1
    max_bins: int = 16
    min_leaf: int = 20
    subsample: float = 1.0
    early_stopping_rounds: Optional[int] = 25
    seed: int = 17
    init: float = 0.0
    stumps: List[Stump] = field(default_factory=list)
    n_features: int = 0
    kind: str = "stumps"

    def _edges(self, X: Sequence[Sequence[float]], feature: int) -> List[float]:
        values = [row[feature] for row in X if row[feature] == row[feature]]
        if not values:
            return []
        cuts = {
            percentile(values, k / self.max_bins) for k in range(1, self.max_bins)
        }
        top = max(values)
        return sorted(c for c in cuts if c < top)

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        *,
        X_val: Optional[Sequence[Sequence[float]]] = None,
        y_val: Optional[Sequence[float]] = None,
    ) -> "StumpQuantileBooster":
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and the same length")
        n = len(X)
        self.n_features = len(X[0])
        self.init = percentile(list(y), self.alpha)
        self.stumps = []
        rng = random.Random(self.seed)

        edges = [self._edges(X, f) for f in range(self.n_features)]
        bins = [
            [
                0 if row[f] != row[f] else bisect.bisect_left(edges[f], row[f])
                for row in X
            ]
            for f in range(self.n_features)
        ]
        fitted = [self.init] * n
        val_fitted = [self.init] * len(X_val) if X_val else []
        best_val, best_round, since_best = float("inf"), 0, 0

        for _ in range(self.n_rounds):
            rows = (
                [i for i in range(n) if rng.random() < self.subsample]
                if self.subsample < 1.0
                else list(range(n))
            )
            if len(rows) < 2 * self.min_leaf:
                rows = list(range(n))
            grad = [self.alpha - (1.0 if y[i] < fitted[i] else 0.0) for i in range(n)]
            total_s = sum(grad[i] for i in rows)
            total_n = len(rows)

            best: Optional[Tuple[float, int, int]] = None
            for f in range(self.n_features):
                width = len(edges[f]) + 1
                if width < 2:
                    continue
                sums = [0.0] * width
                counts = [0] * width
                column = bins[f]
                for i in rows:
                    b = column[i]
                    sums[b] += grad[i]
                    counts[b] += 1
                cum_s, cum_n = 0.0, 0
                for k in range(width - 1):
                    cum_s += sums[k]
                    cum_n += counts[k]
                    right_n = total_n - cum_n
                    if cum_n < self.min_leaf or right_n < self.min_leaf:
                        continue
                    gain = cum_s * cum_s / cum_n + (total_s - cum_s) ** 2 / right_n
                    if best is None or gain > best[0]:
                        best = (gain, f, k)
            if best is None:
                break

            _, f, k = best
            left_res = [y[i] - fitted[i] for i in rows if bins[f][i] <= k]
            right_res = [y[i] - fitted[i] for i in rows if bins[f][i] > k]
            stump = Stump(
                feature=f,
                threshold=edges[f][k],
                left=self.learning_rate * percentile(left_res, self.alpha),
                right=self.learning_rate * percentile(right_res, self.alpha),
            )
            self.stumps.append(stump)
            for i in range(n):
                fitted[i] += stump.left if bins[f][i] <= k else stump.right

            if X_val and y_val is not None and self.early_stopping_rounds:
                for i, row in enumerate(X_val):
                    val_fitted[i] += stump.value(row)
                loss = mean_pinball_loss(y_val, val_fitted, self.alpha)
                if loss < best_val - 1e-9:
                    best_val, best_round, since_best = loss, len(self.stumps), 0
                else:
                    since_best += 1
                    if since_best >= self.early_stopping_rounds:
                        break
        if X_val and y_val is not None and self.early_stopping_rounds and best_round:
            self.stumps = self.stumps[:best_round]
        return self

    def predict_one(self, row: Sequence[float]) -> float:
        return self.init + sum(s.value(row) for s in self.stumps)

    def predict(self, rows: Sequence[Sequence[float]]) -> List[float]:
        return [self.predict_one(row) for row in rows]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "alpha": self.alpha,
            "init": self.init,
            "learning_rate": self.learning_rate,
            "n_features": self.n_features,
            "stumps": [[s.feature, s.threshold, s.left, s.right] for s in self.stumps],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StumpQuantileBooster":
        model = cls(
            alpha=float(data["alpha"]),
            learning_rate=float(data.get("learning_rate", 0.1)),
            init=float(data["init"]),
            n_features=int(data.get("n_features", 0)),
        )
        model.stumps = [
            Stump(int(f), float(t), float(left), float(right))
            for f, t, left, right in data.get("stumps", [])
        ]
        return model

    def save(self, directory: str, stem: str) -> Dict[str, Any]:
        filename = f"{stem}.json"
        with open(os.path.join(directory, filename), "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, separators=(",", ":"))
        return {"kind": self.kind, "alpha": self.alpha, "file": filename}


# --------------------------------------------------------------------------------------
# LightGBM backend
# --------------------------------------------------------------------------------------


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def lightgbm_params(cfg: ModelConfig) -> Dict[str, Any]:
    return {
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "n_estimators": cfg.n_estimators,
        "min_child_samples": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "subsample_freq": cfg.subsample_freq,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_lambda": cfg.reg_lambda,
        "random_state": cfg.random_state,
        "verbose": -1,
    }


class LightGbmQuantile:
    kind = "lightgbm"

    def __init__(
        self,
        alpha: float,
        *,
        params: Optional[Mapping[str, Any]] = None,
        categorical: Sequence[int] = (),
        early_stopping_rounds: Optional[int] = 50,
        booster: Any = None,
    ):
        self.alpha = alpha
        self.params = dict(params or {})
        self.categorical = list(categorical)
        self.early_stopping_rounds = early_stopping_rounds
        self._booster = booster

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        *,
        X_val: Optional[Sequence[Sequence[float]]] = None,
        y_val: Optional[Sequence[float]] = None,
    ) -> "LightGbmQuantile":
        import lightgbm as lgb
        import numpy as np

        model = lgb.LGBMRegressor(objective="quantile", alpha=self.alpha, **self.params)
        kwargs: Dict[str, Any] = {
            "categorical_feature": self.categorical if self.categorical else "auto",
        }
        if X_val is not None and y_val is not None and len(X_val):
            kwargs["eval_set"] = [(np.asarray(X_val, dtype=float), np.asarray(y_val, dtype=float))]
            kwargs["eval_metric"] = "quantile"
            if self.early_stopping_rounds:
                kwargs["callbacks"] = [
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                ]
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float), **kwargs)
        self._booster = model.booster_
        return self

    def predict(self, rows: Sequence[Sequence[float]]) -> List[float]:
        import numpy as np

        if self._booster is None:
            raise RuntimeError("model is not fitted")
        best = getattr(self._booster, "best_iteration", 0) or None
        out = self._booster.predict(np.asarray(rows, dtype=float), num_iteration=best)
        return [float(v) for v in out]

    def save(self, directory: str, stem: str) -> Dict[str, Any]:
        if self._booster is None:
            raise RuntimeError("model is not fitted")
        filename = f"{stem}.lgb"
        self._booster.save_model(os.path.join(directory, filename))
        return {"kind": self.kind, "alpha": self.alpha, "file": filename}

    @classmethod
    def load(cls, path: str, alpha: float) -> "LightGbmQuantile":
        import lightgbm as lgb

        return cls(alpha, booster=lgb.Booster(model_file=path))


# --------------------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------------------


def resolve_backend(requested: str) -> str:
    if requested == "auto":
        return "lightgbm" if lightgbm_available() else "stumps"
    if requested == "lightgbm" and not lightgbm_available():
        raise RuntimeError("model.backend=lightgbm but lightgbm/numpy are not installed")
    return requested


def make_quantile_model(
    alpha: float,
    cfg: ModelConfig,
    *,
    categorical: Sequence[int] = (),
    backend: Optional[str] = None,
) -> Any:
    chosen = resolve_backend(backend or cfg.backend)
    if chosen == "lightgbm":
        return LightGbmQuantile(
            alpha,
            params=lightgbm_params(cfg),
            categorical=categorical,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
    return StumpQuantileBooster(
        alpha=alpha,
        n_rounds=cfg.stump_rounds,
        learning_rate=cfg.stump_learning_rate,
        min_leaf=max(5, cfg.min_child_samples // 2),
        subsample=cfg.subsample,
        early_stopping_rounds=cfg.early_stopping_rounds,
        seed=cfg.random_state,
    )


def load_quantile_model(meta: Mapping[str, Any], directory: str) -> Any:
    path = os.path.join(directory, str(meta["file"]))
    alpha = float(meta["alpha"])
    if meta["kind"] == "lightgbm":
        return LightGbmQuantile.load(path, alpha)
    if meta["kind"] == "stumps":
        with open(path, "r", encoding="utf-8") as handle:
            return StumpQuantileBooster.from_dict(json.load(handle))
    raise ValueError(f"unknown quantile model kind {meta['kind']!r}")


# --------------------------------------------------------------------------------------
# shrunk empirical quantile priors (prep time, dwell)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantilePrior:
    """An empirical p50/p90 with the sample count behind it."""

    p50: float
    p90: float
    n: int = 0

    def to_list(self) -> List[float]:
        return [self.p50, self.p90, float(self.n)]

    @classmethod
    def from_list(cls, data: Sequence[float]) -> "QuantilePrior":
        return cls(float(data[0]), float(data[1]), int(data[2]))


def empirical_prior(values: Sequence[float], upper: float = 0.9) -> Optional[QuantilePrior]:
    if not values:
        return None
    return QuantilePrior(percentile(values, 0.5), percentile(values, upper), len(values))


def shrink_prior(raw: Optional[QuantilePrior], parent: QuantilePrior, k: float) -> QuantilePrior:
    """Blend each quantile toward the parent's by ``n / (n + k)``.

    The quantiles are shrunk independently and then re-ordered, because blending two
    valid (p50 <= p90) pairs with one weight cannot cross but rounding can.
    """
    if raw is None or raw.n <= 0:
        return QuantilePrior(parent.p50, parent.p90, 0)
    w = raw.n / (raw.n + k)
    p50 = w * raw.p50 + (1.0 - w) * parent.p50
    p90 = w * raw.p90 + (1.0 - w) * parent.p90
    return QuantilePrior(p50, max(p50, p90), raw.n)


def alpha_key(alpha: float) -> str:
    """``0.9 -> 'q90'``, used for file stems and manifest keys."""
    return f"q{int(round(alpha * 100)):02d}"


__all__ = [
    "pinball_loss",
    "mean_pinball_loss",
    "coverage",
    "crossing_rate",
    "enforce_monotone",
    "QuantileModel",
    "Stump",
    "StumpQuantileBooster",
    "LightGbmQuantile",
    "lightgbm_available",
    "lightgbm_params",
    "resolve_backend",
    "make_quantile_model",
    "load_quantile_model",
    "alpha_key",
    "QuantilePrior",
    "empirical_prior",
    "shrink_prior",
]
