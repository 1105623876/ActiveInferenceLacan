"""Canonical paths for the reorganized ActiveInferenceLacan repository."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
OUTPUT_ROOT = REPO_ROOT / "outputs"
DOCS_ROOT = REPO_ROOT / "docs"
PAPER_ROOT = REPO_ROOT / "paper"


def output_dir(name: str) -> Path:
    """Return (and create) the output directory for one experiment."""

    path = OUTPUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path
