"""Add scripts/ and charmpheno source to sys.path.

- scripts/   — so `import migrate_checkpoint_drop_gamma` resolves.
- charmpheno/ — so `from charmpheno.export ...` resolves without requiring
  the charmpheno package to be installed into the venv (it's a local path
  dependency in the monorepo, installed as a namespace package whose inner
  sub-package directory must be explicitly on sys.path when running pytest
  from the repo root).
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# scripts/ directory
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# charmpheno source: charmpheno/charmpheno is the actual package directory;
# its parent charmpheno/ must be on sys.path so `import charmpheno` resolves
# to the full package (not the namespace stub from the venv site-packages).
_CHARMPHENO_SRC = _REPO_ROOT / "charmpheno"
if str(_CHARMPHENO_SRC) not in sys.path:
    sys.path.insert(0, str(_CHARMPHENO_SRC))
