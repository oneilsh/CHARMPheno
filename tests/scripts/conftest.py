"""Fixtures for script-level tests."""
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Make scripts/ importable in tests so we can call functions directly.
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    """Run the test from inside a fresh tmp dir so relative-path defaults work."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
