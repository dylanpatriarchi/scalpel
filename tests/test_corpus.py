"""Corpus loading."""

from __future__ import annotations

from pathlib import Path

from scalpel.corpus import load_corpus


def test_bundled_corpus_loads() -> None:
    snippets = load_corpus()
    assert len(snippets) > 20
    # Comments and blank lines are stripped.
    assert all(s and not s.startswith("#") for s in snippets)
    # The sample corpus is clustered by concept, including dogs.
    assert any("dog" in s.lower() for s in snippets)


def test_custom_corpus_parsing(tmp_path: Path) -> None:
    p = tmp_path / "c.txt"
    p.write_text("# header comment\n\nfirst line\n   \nsecond line\n# trailing\n")
    snippets = load_corpus(p)
    assert snippets == ["first line", "second line"]
