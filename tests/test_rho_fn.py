"""Tests for the fixed rho_fn feature."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import pytest

from models.NormalizingFlowFactories import FixedFnConditionalLogDensity_UC


def _bivariate_normal_log_prob(z1, z2, rho):
    """Reference implementation of bivariate normal log-prob."""
    det = 1.0 - rho ** 2
    quad = (z1 ** 2 - 2.0 * rho * z1 * z2 + z2 ** 2) / det
    return -0.5 * (quad + math.log(det) + 2.0 * math.log(2.0 * math.pi))


def test_fixed_rho_fn_matches_analytical_positive_rho():
    """Log-density with constant rho_fn matches closed-form bivariate normal."""
    rho_fn = lambda x: torch.full_like(x, 0.5)
    density = FixedFnConditionalLogDensity_UC(rho_fn)

    z = torch.tensor([[1.0, -0.5], [0.3, 0.8]])
    ctx = torch.zeros(2, 1)
    log_probs = density(z, ctx)

    for i in range(2):
        expected = _bivariate_normal_log_prob(z[i, 0].item(), z[i, 1].item(), rho=0.5)
        assert abs(log_probs[i].item() - expected) < 1e-5, (
            f"Sample {i}: got {log_probs[i].item():.6f}, expected {expected:.6f}"
        )


def test_fixed_rho_fn_per_sample_rho():
    """Different context values produce different rho and thus different log-densities."""
    rho_fn = lambda x: torch.where(x < 0, torch.full_like(x, -0.5), torch.full_like(x, 0.3))
    density = FixedFnConditionalLogDensity_UC(rho_fn)

    z = torch.tensor([[1.0, 1.0], [1.0, 1.0]])          # same z
    ctx = torch.tensor([[-1.0], [1.0]])                   # negative → rho=-0.5, positive → rho=0.3
    log_probs = density(z, ctx)

    expected_neg = _bivariate_normal_log_prob(1.0, 1.0, rho=-0.5)
    expected_pos = _bivariate_normal_log_prob(1.0, 1.0, rho=0.3)

    assert abs(log_probs[0].item() - expected_neg) < 1e-5
    assert abs(log_probs[1].item() - expected_pos) < 1e-5
    assert log_probs[0].item() != log_probs[1].item()


def test_fixed_rho_fn_output_shape():
    """Output is [B] (one scalar per sample)."""
    rho_fn = lambda x: torch.zeros_like(x)
    density = FixedFnConditionalLogDensity_UC(rho_fn)

    z = torch.randn(16, 2)
    ctx = torch.randn(16, 1)
    out = density(z, ctx)

    assert out.shape == (16,), f"Expected (16,), got {out.shape}"


def test_build_rho_fn_gnf_has_correct_density_type():
    """build_rho_fn_gnf produces a model whose z_log_density is FixedFnConditionalLogDensity_UC."""
    from experiments.utils import build_rho_fn_gnf, compute_normalisation, split_data

    rho_fn = lambda x: torch.full_like(x, 0.3)
    data_mu = torch.zeros(2)
    data_sigma = torch.ones(2)

    model = build_rho_fn_gnf(
        rho_fn=rho_fn,
        emb_net=[20, 15, 10],
        int_net=[15, 10, 5],
        nb_steps=10,
        solver="CC",
        l1=0.5,
        nb_flow=1,
        data_mu=data_mu,
        data_sigma=data_sigma,
        nb_epoch_update=50,
        device="cpu",
    )

    assert isinstance(model.z_log_density, FixedFnConditionalLogDensity_UC), (
        f"Expected FixedFnConditionalLogDensity_UC, got {type(model.z_log_density)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
