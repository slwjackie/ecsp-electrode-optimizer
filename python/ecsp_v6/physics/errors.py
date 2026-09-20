"""Typed numerical failures that are confined to selected batch lanes."""

from __future__ import annotations

from typing import Sequence


class BCCandidateBatchError(RuntimeError):
    """A numerical/physical failure confined to named B/C batch lanes."""

    def __init__(
        self,
        message: str,
        candidate_indices: Sequence[int],
        category: str,
    ) -> None:
        indices = tuple(sorted({int(index) for index in candidate_indices}))
        if not indices:
            raise ValueError("BCCandidateBatchError requires candidate indices")
        if not category:
            raise ValueError("BCCandidateBatchError requires a category")
        super().__init__(message)
        self.candidate_indices = indices
        self.category = str(category)

    def __reduce__(self):
        # BaseException otherwise reconstructs subclasses from ``args`` alone,
        # losing structured metadata when CPU worker processes return errors.
        return type(self), (str(self), self.candidate_indices, self.category)
