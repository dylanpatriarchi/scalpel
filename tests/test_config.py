"""Config loading, validation, device resolution, hook-name defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scalpel import config as cfgmod
from scalpel.config import (
    Backend,
    ScalpelConfig,
    load_config,
    mock_config,
    resolve_device,
)

VALID_YAML = """
seed: 3
model:
  name: gpt2
  backend: transformerlens
  device: cpu
sae:
  release: gpt2-small-res-jb
  sae_id: blocks.7.hook_resid_pre
  layer: 7
"""


def test_load_valid(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(VALID_YAML)
    cfg = load_config(p)
    assert cfg.seed == 3
    assert cfg.model.name == "gpt2"
    assert cfg.sae.layer == 7
    # Defaults fill in the optional sections.
    assert cfg.steer.coefs[0] == pytest.approx(-8.0)
    assert cfg.eval.judge_model == "qwen2.5:7b"


def test_hook_name_default() -> None:
    cfg = mock_config()
    cfg2 = ScalpelConfig(
        model=cfg.model,
        sae=cfgmod.SAECfg(release="r", sae_id="s", layer=9),
    )
    assert cfg2.sae.resolved_hook_name == "blocks.9.hook_resid_post"


def test_hook_name_explicit() -> None:
    sae = cfgmod.SAECfg(release="r", sae_id="s", layer=7, hook_name="blocks.7.hook_resid_pre")
    assert sae.resolved_hook_name == "blocks.7.hook_resid_pre"


def test_missing_required_field_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("seed: 1\nmodel:\n  name: gpt2\n")  # no sae
    with pytest.raises(ValidationError):
        load_config(p)


def test_unknown_field_forbidden(tmp_path: Path) -> None:
    p = tmp_path / "extra.yaml"
    p.write_text(VALID_YAML + "\nnonsense: 1\n")
    with pytest.raises(ValidationError):
        load_config(p)


def test_empty_coefs_rejected() -> None:
    with pytest.raises(ValidationError):
        cfgmod.SteerCfg(coefs=[])


def test_negative_layer_rejected() -> None:
    with pytest.raises(ValidationError):
        cfgmod.SAECfg(release="r", sae_id="s", layer=-1)


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_non_mapping_yaml(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_resolve_device_explicit() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_prefers_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_mock_config_is_valid() -> None:
    cfg = mock_config(seed=5)
    assert cfg.model.backend == Backend.mock
    assert cfg.seed == 5
    assert cfg.model.resolved_device == "cpu"
