"""Topic-model evaluation: NPMI coherence on held-out data.

Public API lands incrementally as plan tasks complete. See
docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md.
"""
from spark_vi.eval.topic.types import CoherenceReport

__all__ = ["CoherenceReport"]
