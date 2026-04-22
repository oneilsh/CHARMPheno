"""Training-loop diagnostics: ELBO tracking, live display, checkpointing."""
from spark_vi.diagnostics.checkpoint import load_checkpoint, save_checkpoint

__all__ = ["load_checkpoint", "save_checkpoint"]
