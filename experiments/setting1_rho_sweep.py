"""
Setting 1 — ρ sweep: RMSE vs assumed ρ when true copula is Clayton.

Same structure as Setting 0 ρ sweep, but the data come from a Clayton copula
with standard normal margins.  The rho-GNF model (Gaussian noise assumption)
is trained with a range of *assumed* ρ values while the true DGP stays fixed.

This isolates the effect of copula misspecification: the model can never be
exactly correct because the true copula is not Gaussian, but we ask at which
assumed ρ it best recovers the true ATE.

DGP:
    (u_A, u_Y) ~ Clayton(θ = TRUE_THETA)    [Clayton copula, upper tail dep.]
    e_A = Φ⁻¹(u_A),  e_Y = Φ⁻¹(u_Y)       [standard normal margins]
    A = e_A
    Y = ALPHA * A + e_Y
    True ATE = ALPHA = 0.2

Reference lines on the ρ-curve plot:
    - REF_RHO  ≈ sin(π·τ/2)  where τ = θ/(θ+2)  — Gaussian-copula rho with
                                the same Kendall's τ as the Clayton DGP
    - TRUE ATE = 0.2

Output
------
    results/setting1_rho_sweep/results.npz        — all estimates and errors
    results/setting1_rho_sweep/rho_curve.png      — RMSE + ρ-curve plot
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm as scipy_norm

from experiments.utils import (
    get_cov_matrix,
    split_data,
    compute_normalisation,
    build_rho_gnf,
    train_model,
    estimate_ate,
)

# ── DGP parameters ────────────────────────────────────────────────────────────
TRUE_ATE   = 0.2
ALPHA      = 0.2
TRUE_THETA = 2.0                                      # Clayton copula parameter
TRUE_TAU   = TRUE_THETA / (TRUE_THETA + 2)            # Kendall's τ = 0.5
# Pearson rho of Gaussian copula with the same Kendall's τ (reference line)
REF_RHO    = float(np.sin(np.pi * TRUE_TAU / 2))     # ≈ 0.707

# ── Sweep parameters ──────────────────────────────────────────────────────────
# 20 assumed rho values from -1 to +1, excluding exact boundaries
RHO_VALUES = np.linspace(-0.99, 0.99, 10)
N_SEEDS    = 5
N_SAMPLES  = 5_000

# ── Model / training config ───────────────────────────────────────────────────
CONFIG = dict(
    emb_net          = [20, 15, 10],
    int_net          = [15, 10, 5],
    nb_steps         = 50,
    solver           = "CC",
    nb_flow          = 1,
    l1               = 0.5,
    b_size           = 128,
    nb_epoch         = 20,
    learning_rate    = 3e-4,
    nb_estop         = 20,
    nb_epoch_update  = 50,
    n_mce_samples    = 2_000,
    mce_b_size       = 2_000,
)

RESULTS_DIR = Path(__file__).parent / "results" / "setting1_rho_sweep"


# ── Data generation ───────────────────────────────────────────────────────────

def sample_clayton_copula(n, theta, rng):
    """
    Sample n pairs (u1, u2) from Clayton(theta) via conditional-CDF inversion.
    theta > 0 required (positive dependence / upper tail).
    """
    u1 = rng.uniform(0, 1, n)
    v  = rng.uniform(0, 1, n)
    # Conditional CDF inversion: u2 = (u1^{-θ}(v^{-θ/(θ+1)} - 1) + 1)^{-1/θ}
    inner = u1 ** (-theta) * (v ** (-theta / (theta + 1)) - 1) + 1
    u2 = np.clip(inner ** (-1.0 / theta), 1e-7, 1 - 1e-7)
    return u1, u2


def generate_data(n_samples, alpha, theta, seed):
    """
    Generate (A, Y) from a Clayton copula with standard normal margins.
    Always uses TRUE_THETA regardless of what assumed rho is passed to the model.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    u1, u2 = sample_clayton_copula(n_samples, theta, rng)
    e_A = scipy_norm.ppf(u1).astype(np.float32)
    e_Y = scipy_norm.ppf(u2).astype(np.float32)

    A = torch.from_numpy(e_A).unsqueeze(1)
    Y = alpha * A + torch.from_numpy(e_Y).unsqueeze(1)
    return torch.cat([A, Y], dim=1)


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(assumed_rho, seed):
    """
    Generate data from Clayton DGP, train with assumed_rho (Gaussian), return ATE.
    The mismatch in copula family is the misspecification under test.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data                 = generate_data(N_SAMPLES, ALPHA, TRUE_THETA, seed)
    data_train, data_val = split_data(data)
    data_mu, data_sigma  = compute_normalisation(data_train, data_val)

    Z_Sigma_assumed = get_cov_matrix(assumed_rho)

    model = build_rho_gnf(
        Z_Sigma         = Z_Sigma_assumed,
        emb_net         = CONFIG["emb_net"],
        int_net         = CONFIG["int_net"],
        nb_steps        = CONFIG["nb_steps"],
        solver          = CONFIG["solver"],
        l1              = CONFIG["l1"],
        nb_flow         = CONFIG["nb_flow"],
        data_mu         = data_mu,
        data_sigma      = data_sigma,
        nb_epoch_update = CONFIG["nb_epoch_update"],
        device          = device,
    )
    model, _ = train_model(
        model, data_train, data_val,
        nb_epoch      = CONFIG["nb_epoch"],
        b_size        = CONFIG["b_size"],
        nb_steps      = CONFIG["nb_steps"],
        learning_rate = CONFIG["learning_rate"],
        nb_estop      = CONFIG["nb_estop"],
        device        = device,
    )
    return estimate_ate(
        model,
        Z_Sigma       = Z_Sigma_assumed,
        n_mce_samples = CONFIG["n_mce_samples"],
        mce_b_size    = CONFIG["mce_b_size"],
        device        = device,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(rho_values, all_estimates, save_path):
    """
    Two-panel figure:
      Left  — ρ-curve: mean estimated ATE ± std as a function of assumed ρ
      Right — RMSE vs assumed ρ
    """
    mean_ests = all_estimates.mean(axis=1)
    std_ests  = all_estimates.std(axis=1)
    errors    = all_estimates - TRUE_ATE
    rmses     = np.sqrt((errors ** 2).mean(axis=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: ρ-curve ─────────────────────────────────────────────────────────
    ax1.plot(rho_values, mean_ests, "o-", color="darkorange",
             linewidth=2, markersize=5, label="Mean ATE estimate")
    ax1.fill_between(rho_values,
                     mean_ests - std_ests,
                     mean_ests + std_ests,
                     alpha=0.2, color="darkorange", label="±1 std (across seeds)")
    ax1.axhline(TRUE_ATE, color="red",   linestyle="--", linewidth=1.5,
                label=f"True ATE = {TRUE_ATE}")
    ax1.axvline(REF_RHO,  color="green", linestyle="--", linewidth=1.5,
                label=f"Gaussian-equiv ρ ≈ {REF_RHO:.3f}  (τ={TRUE_TAU:.2f})")

    ax1.set_xlabel("Assumed ρ (Gaussian model)")
    ax1.set_ylabel("Estimated ATE")
    ax1.set_title(
        f"ρ-curve  (Clayton θ={TRUE_THETA}, τ={TRUE_TAU:.2f}, true ATE={TRUE_ATE})\n"
        f"{N_SEEDS} seeds · {CONFIG['nb_epoch']} epochs · n={N_SAMPLES:,}"
    )
    ax1.legend(fontsize=8)

    # ── Right: RMSE vs assumed ρ ──────────────────────────────────────────────
    ax2.plot(rho_values, rmses, "s-", color="steelblue",
             linewidth=2, markersize=5, label="RMSE")
    ax2.axvline(REF_RHO, color="green", linestyle="--", linewidth=1.5,
                label=f"Gaussian-equiv ρ ≈ {REF_RHO:.3f}")

    best_idx = int(np.argmin(rmses))
    ax2.scatter(rho_values[best_idx], rmses[best_idx],
                color="red", zorder=5, s=80,
                label=f"Min RMSE at ρ = {rho_values[best_idx]:.2f}")

    ax2.set_xlabel("Assumed ρ (Gaussian model)")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE vs Assumed ρ\n(Clayton DGP — Gaussian model misspecification)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Setting 1 ρ sweep — Clayton θ={TRUE_THETA}, τ={TRUE_TAU:.3f}, "
          f"Gaussian-equiv ref ρ={REF_RHO:.4f}")

    all_estimates = np.zeros((len(RHO_VALUES), N_SEEDS))

    for i, rho in enumerate(RHO_VALUES):
        for s in range(N_SEEDS):
            print(f"[assumed_rho={rho:+.3f}  seed={s}]", end="  ", flush=True)
            ate = run_single(assumed_rho=rho, seed=s)
            all_estimates[i, s] = ate
            print(f"ATE_est={ate:+.4f}  err={ate - TRUE_ATE:+.4f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== ρ Sweep Summary  (Clayton θ={TRUE_THETA}, true ATE={TRUE_ATE}) ===")
    print(f"{'Assumed ρ':>10}  {'Mean est':>10}  {'Bias':>8}  {'RMSE':>8}")
    for i, rho in enumerate(RHO_VALUES):
        ests   = all_estimates[i]
        bias   = np.mean(ests - TRUE_ATE)
        rmse   = np.sqrt(np.mean((ests - TRUE_ATE) ** 2))
        marker = " ← Gaussian-equiv ρ" if abs(rho - REF_RHO) == min(abs(RHO_VALUES - REF_RHO)) else ""
        print(f"{rho:>10.3f}  {np.mean(ests):>10.4f}  {bias:>8.4f}  {rmse:>8.4f}{marker}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        RESULTS_DIR / "results.npz",
        rho_values    = RHO_VALUES,
        all_estimates = all_estimates,
        true_theta    = TRUE_THETA,
        true_tau      = TRUE_TAU,
        ref_rho       = REF_RHO,
        true_ate      = TRUE_ATE,
    )

    plot_results(RHO_VALUES, all_estimates, RESULTS_DIR / "rho_curve.png")


if __name__ == "__main__":
    main()
