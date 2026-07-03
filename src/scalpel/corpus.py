"""Corpus loading for feature discovery.

A corpus is a flat list of short text snippets (one per line). We ship a small
illustrative corpus, clustered by concept, so ``scalpel discover`` works out of
the box; pass ``--corpus path`` to use your own.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

DEFAULT_CORPUS = "corpus_sample.txt"


def _parse(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_corpus(path: str | Path | None = None) -> list[str]:
    """Load corpus snippets from ``path`` or the bundled sample.

    Blank lines and lines starting with ``#`` are ignored.
    """
    if path is not None:
        return _parse(Path(path).read_text(encoding="utf-8"))
    text = (files("scalpel.data") / DEFAULT_CORPUS).read_text(encoding="utf-8")
    return _parse(text)
