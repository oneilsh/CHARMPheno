"""TopicBlockPartition: the gating layout for OnlineSTM background/foreground blocks.

Partitions the K topics into a background block (every document may express it)
and one contiguous foreground block per group (only that group's documents may
express it). The engine consumes only the resolved index sets; the contiguous
layout (background first, then groups in declared order) is a readability
convention. See docs/superpowers/specs/2026-06-23-gated-stm-background-foreground-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopicBlockPartition:
    group_var: str
    background_k: int
    foreground: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.background_k < 1:
            raise ValueError(f"background_k must be >= 1, got {self.background_k}")
        labels = [g for g, _ in self.foreground]
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate group labels in foreground: {labels}")
        for g, size in self.foreground:
            if size < 1:
                raise ValueError(f"foreground block '{g}' size must be >= 1, got {size}")

    @property
    def K(self) -> int:
        return self.background_k + sum(size for _, size in self.foreground)

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(g for g, _ in self.foreground)

    def background_indices(self) -> np.ndarray:
        return np.arange(self.background_k, dtype=np.int64)

    def _block_start(self, group: str) -> int:
        start = self.background_k
        for g, size in self.foreground:
            if g == group:
                return start
            start += size
        raise KeyError(f"unknown group {group!r}; known groups: {self.groups}")

    def block_indices(self, group: str) -> np.ndarray:
        start = self._block_start(group)
        size = dict(self.foreground)[group]
        return np.arange(start, start + size, dtype=np.int64)

    def allowed_indices(self, groups: frozenset[str]) -> np.ndarray:
        # A group with no foreground block contributes nothing (background-only).
        # This is what lets a large "common" cohort inform the background while
        # only rare groups carry foreground topics.
        known = set(self.groups)
        parts = [self.background_indices()]
        for g in sorted(groups):
            if g in known:
                parts.append(self.block_indices(g))
        return np.unique(np.concatenate(parts)).astype(np.int64)

    def topic_labels(self) -> list[str]:
        labels = ["background"] * self.background_k
        for g, size in self.foreground:
            labels.extend([g] * size)
        return labels

    def to_dict(self) -> dict:
        return {
            "group_var": self.group_var,
            "background_k": self.background_k,
            "foreground": [[g, size] for g, size in self.foreground],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopicBlockPartition":
        return cls(
            group_var=d["group_var"],
            background_k=int(d["background_k"]),
            foreground=tuple((g, int(size)) for g, size in d["foreground"]),
        )
