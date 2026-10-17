"""The merchant prep-time model: the ``prep`` term of ``prep + travel + handoff``.

Prep time is not a routing problem. It is a per-kitchen queueing problem, and it is
usually the largest and most variable term in a delivery ETA. Two layers:

1. **Priors.** Every merchant gets an empirical p50/p90 of its historical prep, shrunk
   toward its cuisine category's, which is shrunk toward the global one — the same
   credibility rule as the speed profiles, for the same reason. A merchant onboarded
   yesterday with two orders inherits "pizza places take 14 minutes" rather than
   "this place takes exactly 6 minutes, twice".
2. **Corrections.** A quantile model predicts the residual against the merchant's
   prior from what the prior cannot see: item count, order value, hour of day, and
   the live ``open_orders_at_merchant`` load signal that separates a slow kitchen from
   a slammed one.

The corrections are trained leak-free by fitting the priors on the *earlier* part of
the training window and the correction model on the later part, so a training row's
prior never contains its own prep time. For serving, the priors are then refitted on
the whole training window.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from statistics import fmean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..features.calendar import localize
from ..features.spec import FeatureSpec
from ..types import PrepObservation, QuantilePair
from .quantile import (
    QuantilePrior,
    alpha_key,
    coverage,
    empirical_prior,
    load_quantile_model,
    make_quantile_model,
    mean_pinball_loss,
    resolve_backend,
    shrink_prior,
)
from .residual import upper_alpha
from .splits import time_split

PREP_FILE = "prep.json"
MIN_CORRECTION_ROWS = 200

PREP_SPEC = FeatureSpec(
    names=(
        "merchant_p50_s",
        "merchant_p90_s",
        "merchant_log_n",
        "is_cold_start",
        "item_count",
        "order_value_usd",
        "open_orders",
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "cuisine_code",
    ),
    categorical=("cuisine_code",),
    version=1,
)


@dataclass
class PrepPriors:
    global_prior: QuantilePrior
    cuisine: Dict[str, QuantilePrior] = field(default_factory=dict)
    merchant: Dict[str, QuantilePrior] = field(default_factory=dict)
    merchant_cuisine: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def fit(cls, observations: Sequence[PrepObservation], k: float) -> "PrepPriors":
        values = [o.prep_s for o in observations]
        global_prior = empirical_prior(values) or QuantilePrior(600.0, 1020.0, 0)
        by_cuisine: Dict[str, List[float]] = defaultdict(list)
        by_merchant: Dict[str, List[float]] = defaultdict(list)
        merchant_cuisine: Dict[str, str] = {}
        for o in observations:
            by_cuisine[o.cuisine_category].append(o.prep_s)
            by_merchant[o.merchant_id].append(o.prep_s)
            merchant_cuisine[o.merchant_id] = o.cuisine_category
        cuisine = {
            c: shrink_prior(empirical_prior(v), global_prior, k) for c, v in by_cuisine.items()
        }
        merchant = {
            m: shrink_prior(
                empirical_prior(v), cuisine.get(merchant_cuisine[m], global_prior), k
            )
            for m, v in by_merchant.items()
        }
        return cls(global_prior, cuisine, merchant, merchant_cuisine)

    def lookup(self, merchant_id: Optional[str], cuisine: Optional[str]) -> Tuple[QuantilePrior, bool]:
        """``(prior, cold_start)`` for a merchant, falling back cuisine -> global."""
        if merchant_id and merchant_id in self.merchant:
            return self.merchant[merchant_id], False
        category = cuisine or (self.merchant_cuisine.get(merchant_id or "") if merchant_id else None)
        if category and category in self.cuisine:
            return self.cuisine[category], True
        return self.global_prior, True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global": self.global_prior.to_list(),
            "cuisine": {k: v.to_list() for k, v in self.cuisine.items()},
            "merchant": {k: v.to_list() for k, v in self.merchant.items()},
            "merchant_cuisine": dict(self.merchant_cuisine),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PrepPriors":
        return cls(
            global_prior=QuantilePrior.from_list(data["global"]),
            cuisine={k: QuantilePrior.from_list(v) for k, v in data.get("cuisine", {}).items()},
            merchant={k: QuantilePrior.from_list(v) for k, v in data.get("merchant", {}).items()},
            merchant_cuisine=dict(data.get("merchant_cuisine", {})),
        )


def prep_features(
    priors: PrepPriors,
    cuisine_codes: Mapping[str, int],
    *,
    merchant_id: Optional[str],
    cuisine: Optional[str],
    placed_at: datetime,
    item_count: Optional[int],
    order_value_usd: Optional[float],
    open_orders: Optional[int],
    tz: Optional[str],
) -> Tuple[List[float], QuantilePrior]:
    prior, cold = priors.lookup(merchant_id, cuisine)
    category = cuisine or priors.merchant_cuisine.get(merchant_id or "", "other")
    local = localize(placed_at, tz)
    hour = local.hour + local.minute / 60.0
    angle = 2.0 * math.pi * hour / 24.0
    features = {
        "merchant_p50_s": prior.p50,
        "merchant_p90_s": prior.p90,
        "merchant_log_n": math.log1p(prior.n),
        "is_cold_start": 1.0 if cold else 0.0,
        "item_count": float(item_count) if item_count is not None else float("nan"),
        "order_value_usd": float(order_value_usd) if order_value_usd is not None else float("nan"),
        "open_orders": float(open_orders) if open_orders is not None else float("nan"),
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "is_weekend": 1.0 if local.weekday() >= 5 else 0.0,
        "cuisine_code": float(cuisine_codes.get(category, 0)),
    }
    return PREP_SPEC.vectorize(features), prior


class PrepTimeModel:
    def __init__(
        self,
        priors: PrepPriors,
        *,
        cuisine_codes: Optional[Mapping[str, int]] = None,
        models: Optional[Mapping[float, Any]] = None,
        tz: Optional[str] = "America/New_York",
        backend: str = "stumps",
        floor_s: float = 60.0,
    ):
        self.priors = priors
        self.cuisine_codes = dict(cuisine_codes or {})
        self.models = dict(models or {})
        self.tz = tz
        self.backend = backend
        self.floor_s = floor_s

    @property
    def has_corrections(self) -> bool:
        return 0.5 in self.models

    def predict(
        self,
        *,
        merchant_id: Optional[str],
        placed_at: datetime,
        cuisine: Optional[str] = None,
        item_count: Optional[int] = None,
        order_value_usd: Optional[float] = None,
        open_orders: Optional[int] = None,
    ) -> QuantilePair:
        vector, prior = prep_features(
            self.priors,
            self.cuisine_codes,
            merchant_id=merchant_id,
            cuisine=cuisine,
            placed_at=placed_at,
            item_count=item_count,
            order_value_usd=order_value_usd,
            open_orders=open_orders,
            tz=self.tz,
        )
        if not self.has_corrections:
            return QuantilePair(max(self.floor_s, prior.p50), max(self.floor_s, prior.p90))
        hi = upper_alpha(sorted(self.models))
        r50 = self.models[0.5].predict([vector])[0]
        r90 = self.models[hi].predict([vector])[0]
        p50 = max(self.floor_s, prior.p50 + r50)
        return QuantilePair(p50, max(p50, prior.p50 + r90))

    # -- training ----------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        observations: Sequence[PrepObservation],
        cfg: Config,
        *,
        backend: Optional[str] = None,
    ) -> Tuple["PrepTimeModel", Dict[str, Any]]:
        chosen = resolve_backend(backend or cfg.model.backend)
        k = cfg.model.prep_shrink_k
        tz = cfg.profiles.timezone
        split = time_split(
            [o.placed_at for o in observations],
            train_fraction=cfg.model.train_fraction,
            validation_fraction=cfg.model.validation_fraction,
        )
        train = [observations[i] for i in split.train]
        validation = [observations[i] for i in split.validation]
        test = [observations[i] for i in split.test]
        cuisine_codes = {
            c: i + 1 for i, c in enumerate(sorted({o.cuisine_category for o in train}))
        }

        # Priors from the first part of training, corrections fitted on the rest.
        cut = int(len(train) * 0.6)
        prior_part, fit_part = train[:cut], train[cut:]
        models: Dict[float, Any] = {}
        if len(fit_part) >= MIN_CORRECTION_ROWS and prior_part:
            early = PrepPriors.fit(prior_part, k)

            def rows(obs: Sequence[PrepObservation], priors: PrepPriors):
                X, y = [], []
                for o in obs:
                    vector, prior = prep_features(
                        priors, cuisine_codes,
                        merchant_id=o.merchant_id, cuisine=o.cuisine_category,
                        placed_at=o.placed_at, item_count=o.item_count,
                        order_value_usd=o.order_value_usd,
                        open_orders=o.open_orders_at_merchant, tz=tz,
                    )
                    X.append(vector)
                    y.append(o.prep_s - prior.p50)
                return X, y

            X_fit, y_fit = rows(fit_part, early)
            X_val, y_val = rows(validation, early) if validation else (None, None)
            for alpha in cfg.model.quantiles:
                model = make_quantile_model(
                    alpha, cfg.model, categorical=PREP_SPEC.categorical_indices, backend=chosen
                )
                models[alpha] = model.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)

        final = cls(
            PrepPriors.fit(train, k),
            cuisine_codes=cuisine_codes,
            models=models,
            tz=tz,
            backend=chosen,
        )
        report = {
            "sizes": split.sizes(),
            "merchants": len(final.priors.merchant),
            "cuisines": len(final.priors.cuisine),
            "corrections": final.has_corrections,
            "test": final.evaluate(test),
        }
        return final, report

    def evaluate(self, observations: Sequence[PrepObservation]) -> Dict[str, float]:
        if not observations:
            return {}
        predictions = [
            self.predict(
                merchant_id=o.merchant_id, placed_at=o.placed_at, cuisine=o.cuisine_category,
                item_count=o.item_count, order_value_usd=o.order_value_usd,
                open_orders=o.open_orders_at_merchant,
            )
            for o in observations
        ]
        actual = [o.prep_s for o in observations]
        return {
            "rows": float(len(actual)),
            "mae_p50": fmean(abs(a - p.p50) for a, p in zip(actual, predictions)),
            "pinball_p50": mean_pinball_loss(actual, [p.p50 for p in predictions], 0.5),
            "pinball_p90": mean_pinball_loss(actual, [p.p90 for p in predictions], 0.9),
            "coverage_p90": coverage(actual, [p.p90 for p in predictions]),
        }

    # -- persistence -------------------------------------------------------------

    def save(self, directory: str) -> Dict[str, Any]:
        os.makedirs(directory, exist_ok=True)
        entries = [
            m.save(directory, f"prep_{alpha_key(a)}") for a, m in sorted(self.models.items())
        ]
        meta = {
            "kind": "prep",
            "backend": self.backend,
            "tz": self.tz,
            "floor_s": self.floor_s,
            "feature_spec_hash": PREP_SPEC.hash,
            "cuisine_codes": self.cuisine_codes,
            "priors": self.priors.to_dict(),
            "quantiles": entries,
        }
        with open(os.path.join(directory, PREP_FILE), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, separators=(",", ":"), sort_keys=True)
        return {k: v for k, v in meta.items() if k not in ("priors",)}

    @classmethod
    def load(cls, directory: str) -> "PrepTimeModel":
        with open(os.path.join(directory, PREP_FILE), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        PREP_SPEC.check(str(meta.get("feature_spec_hash", "")))
        return cls(
            PrepPriors.from_dict(meta["priors"]),
            cuisine_codes={str(k): int(v) for k, v in meta.get("cuisine_codes", {}).items()},
            models={
                float(e["alpha"]): load_quantile_model(e, directory)
                for e in meta.get("quantiles", [])
            },
            tz=meta.get("tz"),
            backend=str(meta.get("backend", "stumps")),
            floor_s=float(meta.get("floor_s", 60.0)),
        )


__all__ = ["PrepTimeModel", "PrepPriors", "PREP_SPEC", "prep_features", "PREP_FILE"]
