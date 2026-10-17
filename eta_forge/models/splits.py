"""Time-based train / validation / test splitting.

Never random. A random split puts Tuesday's trips in training and Wednesday's in test
when both were drawn from the same congestion regime, the same weather, the same road
closure — and, worst of all, lets the target-encoded cell features of a test row be
computed from training rows that happened *after* it. Offline numbers from a random
split are reliably beautiful and reliably meaningless.

The split here sorts by timestamp and cuts at fractions of the ordered corpus, with an
optional **embargo** gap after each boundary: rows within the gap are dropped, so a
multi-hour incident straddling a boundary cannot leak across it through the encodings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

from ..types import ValidationError


@dataclass
class TimeSplit:
    train: List[int] = field(default_factory=list)
    validation: List[int] = field(default_factory=list)
    test: List[int] = field(default_factory=list)
    train_end: Optional[datetime] = None
    validation_end: Optional[datetime] = None
    embargoed: int = 0

    def sizes(self) -> Dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
            "embargoed": self.embargoed,
        }


def time_split(
    timestamps: Sequence[datetime],
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    embargo: timedelta = timedelta(0),
) -> TimeSplit:
    """Split row positions by time. Each returned list is in time order."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValidationError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValidationError("train + validation fractions must leave a test set")
    n = len(timestamps)
    order = sorted(range(n), key=lambda i: timestamps[i])
    if n < 3:
        return TimeSplit(train=order)

    cut_train = max(1, int(n * train_fraction))
    cut_val = max(cut_train + 1, int(n * (train_fraction + validation_fraction)))
    cut_val = min(cut_val, n - 1)

    train = order[:cut_train]
    validation = order[cut_train:cut_val]
    test = order[cut_val:]
    train_end = timestamps[train[-1]]
    validation_end = timestamps[validation[-1]] if validation else train_end

    embargoed = 0
    if embargo > timedelta(0):
        kept_val = [i for i in validation if timestamps[i] > train_end + embargo]
        kept_test = [i for i in test if timestamps[i] > validation_end + embargo]
        embargoed = (len(validation) - len(kept_val)) + (len(test) - len(kept_test))
        validation, test = kept_val, kept_test

    return TimeSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=train_end,
        validation_end=validation_end,
        embargoed=embargoed,
    )


def time_blocks(ordered: Sequence[int], k: int) -> List[List[int]]:
    """Cut an already time-ordered list of positions into ``k`` contiguous blocks."""
    if k < 1:
        raise ValidationError("k must be >= 1")
    n = len(ordered)
    k = min(k, n) if n else 1
    blocks: List[List[int]] = []
    for b in range(k):
        start = (n * b) // k
        end = (n * (b + 1)) // k
        blocks.append(list(ordered[start:end]))
    return blocks


def assert_no_leakage(split: TimeSplit, timestamps: Sequence[datetime]) -> None:
    """Raise if any validation/test row is not strictly later than every train row."""
    if not split.train:
        return
    latest_train = max(timestamps[i] for i in split.train)
    for name, rows in (("validation", split.validation), ("test", split.test)):
        if rows and min(timestamps[i] for i in rows) < latest_train:
            raise ValidationError(f"{name} rows precede the end of training")
    if split.validation and split.test:
        if min(timestamps[i] for i in split.test) < max(timestamps[i] for i in split.validation):
            raise ValidationError("test rows precede the end of validation")


__all__ = ["TimeSplit", "time_split", "time_blocks", "assert_no_leakage"]
