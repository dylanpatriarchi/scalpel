"""SAE encode/decode shapes, exact reconstruction, feature directions, stats."""

from __future__ import annotations

import pytest
import torch

from scalpel.sae import SAEWrapper


def test_shapes(random_sae: SAEWrapper) -> None:
    x = torch.randn(5, random_sae.d_model)
    features = random_sae.encode(x)
    recon = random_sae.decode(features)
    assert features.shape == (5, random_sae.d_sae)
    assert recon.shape == (5, random_sae.d_model)


def test_d_model_d_sae(random_sae: SAEWrapper) -> None:
    assert random_sae.d_model == 4
    assert random_sae.d_sae == 16


def test_exact_reconstruction_of_nonnegative(
    identity_sae: SAEWrapper, positive_acts: torch.Tensor
) -> None:
    recon = identity_sae.reconstruct(positive_acts)
    assert torch.allclose(recon, positive_acts, atol=1e-6)


def test_relu_zeroes_negatives(identity_sae: SAEWrapper) -> None:
    x = torch.tensor([[-1.0, 2.0, -3.0, 4.0, 0.0, -0.5, 1.0, -2.0]])
    recon = identity_sae.reconstruct(x)
    assert torch.allclose(recon, torch.relu(x), atol=1e-6)


def test_feature_direction_is_decoder_row(random_sae: SAEWrapper) -> None:
    for i in (0, 5, 15):
        assert torch.equal(random_sae.feature_direction(i), random_sae.W_dec[i])


def test_feature_direction_out_of_range(random_sae: SAEWrapper) -> None:
    with pytest.raises(IndexError):
        random_sae.feature_direction(random_sae.d_sae)
    with pytest.raises(IndexError):
        random_sae.feature_direction(-1)


def test_reconstruction_error_matches_manual(random_sae: SAEWrapper) -> None:
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(20, random_sae.d_model, generator=gen)

    stats = random_sae.reconstruction_error(x)

    x_hat = random_sae.decode(random_sae.encode(x))
    resid = x - x_hat
    expected_mse = torch.mean(resid**2)
    total = torch.mean((x - x.mean(dim=0, keepdim=True)) ** 2)
    expected_fve = 1.0 - (expected_mse / (total + 1e-12))

    assert stats.n_tokens == 20
    assert stats.mse == pytest.approx(expected_mse.item(), rel=1e-5)
    assert stats.variance_explained == pytest.approx(expected_fve.item(), rel=1e-5)


def test_identity_sae_perfect_stats(identity_sae: SAEWrapper, positive_acts: torch.Tensor) -> None:
    stats = identity_sae.reconstruction_error(positive_acts)
    assert stats.mse == pytest.approx(0.0, abs=1e-10)
    assert stats.variance_explained == pytest.approx(1.0, abs=1e-6)
    # Every nonnegative dimension is an active latent under the identity SAE.
    assert stats.mean_l0 == pytest.approx(float(identity_sae.d_model))


def test_l0_counts_active_features(identity_sae: SAEWrapper) -> None:
    x = torch.tensor([[1.0, 0.0, -1.0, 2.0, 0.0, 0.0, 3.0, -4.0]])
    stats = identity_sae.reconstruction_error(x)
    # Positive entries: 1.0, 2.0, 3.0 -> L0 == 3.
    assert stats.mean_l0 == pytest.approx(3.0)


def test_1d_input_is_promoted(identity_sae: SAEWrapper) -> None:
    x = torch.rand(8) + 0.1
    stats = identity_sae.reconstruction_error(x)
    assert stats.n_tokens == 1


def test_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="d_sae"):
        SAEWrapper(
            W_enc=torch.zeros(4, 16),
            W_dec=torch.zeros(8, 4),  # d_sae 8 != 16
            b_enc=torch.zeros(16),
            b_dec=torch.zeros(4),
            hook_name="h",
            layer=0,
        )


def test_reconstruct_stats_stable_across_reshape(random_sae: SAEWrapper) -> None:
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(6, random_sae.d_model, generator=gen)
    flat = random_sae.reconstruction_error(x)
    reshaped = random_sae.reconstruction_error(x.reshape(2, 3, random_sae.d_model))
    assert flat.mse == pytest.approx(reshaped.mse, rel=1e-6)
    assert flat.n_tokens == reshaped.n_tokens
