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


def test_discover_mock_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["discover", "--concept", "dog", "--backend", "mock", "--top-k", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "scalpel discover" in out
    assert "top 5 features" in out


def test_discover_no_match_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["discover", "--concept", "zzznotincorpus", "--backend", "mock"])
    assert rc == 2
    assert "No snippet" in capsys.readouterr().err


def test_steer_mock_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["steer", "--feature", "1", "--coef", "3", "--prompt", "hello", "--backend", "mock"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--- unsteered ---" in out
    assert "--- steered ---" in out
    # The mock backend tags steered output with the coefficient.
    assert "coef=3" in out


def test_steer_out_of_range_feature_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["steer", "--feature", "9999", "--coef", "1", "--prompt", "hi", "--backend", "mock"])
    assert rc == 2
    assert "out of range" in capsys.readouterr().err


def test_eval_mock_runs(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "eval",
            "--feature",
            "1",
            "--concept",
            "dog",
            "--backend",
            "mock",
            "--coefs",
            "0",
            "2",
            "4",
            "--no-plots",
            "--out",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "scalpel eval" in out
    assert "coef" in out
    assert (tmp_path / "sweep_sae.csv").exists()
    assert (tmp_path / "sweep_sae.json").exists()


def test_eval_out_of_range_feature_returns_2(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "eval",
            "--feature",
            "9999",
            "--concept",
            "dog",
            "--backend",
            "mock",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 2
    assert "out of range" in capsys.readouterr().err


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
