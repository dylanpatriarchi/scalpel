"""CLI wiring via the download-free mock backend."""

from __future__ import annotations

import pytest

from scalpel.backends import build_backend
from scalpel.cli import build_sae, main
from scalpel.config import mock_config


def test_smoke_mock_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["smoke", "--backend", "mock"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "scalpel smoke" in out
    assert "reconstruction MSE" in out
    assert "variance explained" in out


def test_smoke_mock_reconstructs_exactly(capsys: pytest.CaptureFixture[str]) -> None:
    # Identity mock SAE + nonnegative mock activations -> ~perfect reconstruction.
    main(["smoke", "--backend", "mock"])
    out = capsys.readouterr().out
    assert "variance explained: 1.0000" in out


def test_smoke_respects_seed_override(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["smoke", "--backend", "mock", "--seed", "99"])
    assert rc == 0
    assert "scalpel smoke" in capsys.readouterr().out


def test_build_sae_mock_is_identity() -> None:
    cfg = mock_config()
    backend = build_backend(cfg)
    sae = build_sae(cfg, backend)
    assert sae.d_model == backend.d_model


def test_unimplemented_commands_return_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["discover", "--concept", "dogs"]) == 2
    assert main(["steer", "--feature", "1", "--coef", "2", "--prompt", "hi"]) == 2
    assert main(["eval"]) == 2


def test_missing_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_mock_backend_determinism() -> None:
    cfg = mock_config()
    b1 = build_backend(cfg)
    b2 = build_backend(cfg)
    import torch

    a1 = b1.capture_resid("one two three", "blocks.0.hook_resid_post")
    a2 = b2.capture_resid("one two three", "blocks.0.hook_resid_post")
    assert torch.equal(a1, a2)
