"""Typed configuration for Scalpel runs.

Everything the tool does is driven by a :class:`ScalpelConfig` loaded from YAML,
so the model, SAE, layer, steering sweep and evaluation set are all swappable
without touching code. Nothing about the model or feature is hardcoded.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Backend(StrEnum):
    """Which hooking backend runs the model."""

    transformerlens = "transformerlens"
    nnsight = "nnsight"
    mock = "mock"


def resolve_device(pref: str = "auto") -> str:
    """Resolve a device preference to a concrete device string.

    ``auto`` picks CUDA, then Apple MPS, then CPU. Any explicit value is
    returned unchanged so callers can force a device.
    """
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelCfg(BaseModel):
    """The model to steer and how to run it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="HF / TransformerLens model id, or 'mock'.")
    backend: Backend = Backend.transformerlens
    device: str = "auto"
    dtype: str = "float32"

    @property
    def resolved_device(self) -> str:
        return resolve_device(self.device)


class SAECfg(BaseModel):
    """Which released SAE to load and where it hooks the residual stream."""

    model_config = ConfigDict(extra="forbid")

    release: str = Field(description="SAELens release id, e.g. 'gpt2-small-res-jb'.")
    sae_id: str = Field(description="SAE id within the release.")
    layer: int = Field(ge=0, description="Transformer block the SAE reads from.")
    hook_name: str | None = Field(
        default=None,
        description="Residual hook point; defaults to blocks.{layer}.hook_resid_post.",
    )

    @property
    def resolved_hook_name(self) -> str:
        return self.hook_name or f"blocks.{self.layer}.hook_resid_post"


class SteerCfg(BaseModel):
    """The coefficient sweep applied to a steering vector."""

    model_config = ConfigDict(extra="forbid")

    coefs: list[float] = Field(
        default_factory=lambda: [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0],
        description="Coefficients to sweep, including negatives (suppression).",
    )
    max_new_tokens: int = Field(default=40, gt=0)

    @field_validator("coefs")
    @classmethod
    def _non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("coefs must contain at least one value")
        return v


class EvalCfg(BaseModel):
    """Evaluation prompt set and judge configuration."""

    model_config = ConfigDict(extra="forbid")

    prompts: list[str] = Field(
        default_factory=lambda: [
            "I think that",
            "The weather today",
            "Let me tell you about",
        ]
    )
    judge_model: str = "qwen2.5:7b"
    judge_host: str = "http://localhost:11434"


class ScalpelConfig(BaseModel):
    """Top-level configuration for a run."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    model: ModelCfg
    sae: SAECfg
    steer: SteerCfg = Field(default_factory=SteerCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)

    @model_validator(mode="after")
    def _mock_consistency(self) -> ScalpelConfig:
        # A mock run does not touch the SAE layer, so we allow any placeholder
        # SAE fields, but a real backend must point at a plausible layer.
        return self


def load_config(path: str | Path) -> ScalpelConfig:
    """Load and validate a :class:`ScalpelConfig` from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")
    return ScalpelConfig.model_validate(raw)


def mock_config(seed: int = 0) -> ScalpelConfig:
    """A self-contained config for the download-free mock backend."""
    return ScalpelConfig(
        seed=seed,
        model=ModelCfg(name="mock", backend=Backend.mock, device="cpu"),
        sae=SAECfg(release="mock", sae_id="mock", layer=0, hook_name="blocks.0.hook_resid_post"),
    )
