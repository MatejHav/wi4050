"""
Setting: Piecewise Gaussian Copula — rho changes sign at X = 0.

DGP
---
    X       ~ Uniform(-1, 1)
    Z_X      = Φ⁻¹((X + 1) / 2)           standard-normal score of X
    if X < 0:  Z_Y | Z_X ~ N(RHO_NEG · Z_X, √(1 − RHO_NEG²))
    if X ≥ 0:  Z_Y | Z_X ~ N(RHO_POS · Z_X, √(1 − RHO_POS²))
    Y  = Z_Y                               marginally Y ~ N(0, 1)

The two regimes differ in both magnitude and *sign* of dependence:
    RHO_NEG = −0.5   (negative tail co-movement for X < 0)
    RHO_POS =  0.3   (positive co-movement for X ≥ 0)

Because X is uniform on (−1, 1), the two regimes are equally likely (50 / 50).

True ATE (analytical)
---------------------
    E[Y | do(X = x)] = ρ(x) · Φ⁻¹((x + 1) / 2)

    Reference contrast: x₀ = −0.5  (X<0 regime)  vs  x₁ = 0.5  (X≥0 regime)
    EY0 = RHO_NEG · Φ⁻¹(0.25) ≈ −0.5 · (−0.6745) ≈  0.337
    EY1 = RHO_POS · Φ⁻¹(0.75) ≈  0.3 ·  0.6745   ≈  0.202
    ATE = EY1 − EY0                               ≈ −0.135

Experiment
----------
Sweep the *assumed* ρ that the rho-GNF model uses for its latent reference
distribution over [−0.99, 0.99].  For each assumed ρ, N_SEEDS models are
trained on fresh data and then used to estimate the ATE at the reference
contrast (do(X=−0.5) vs do(X=0.5)).

Output
------
    results/setting_piecewise_copula/results.npz   — raw estimates + errors
    results/setting_piecewise_copula/ate_figure.png — ρ-curve + RMSE figure
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
RHO_NEG = -0.5   # copula rho when X < 0
RHO_POS =  0.3   # copula rho when X >= 0

# Reference intervention contrast for ATE
X0 = -0.5        # treatment value in the negative regime
X1 =  0.5        # treatment value in the positive regime

# Analytical true values
_u0    = np.clip((X0 + 1) / 2, 1e-6, 1 - 1e-6)
_u1    = np.clip((X1 + 1) / 2, 1e-6, 1 - 1e-6)
TRUE_EY0 = float(RHO_NEG * scipy_norm.ppf(_u0))   # ≈  0.337
TRUE_EY1 = float(RHO_POS * scipy_norm.ppf(_u1))   # ≈  0.202
TRUE_ATE  = TRUE_EY1 - TRUE_EY0                    # ≈ −0.135

# ── Sweep / experiment parameters ─────────────────────────────────────────────
RHO_VALUES = np.linspace(-0.99, 0.99, 20)
N_SEEDS    = 5
N_SAMPLES  = 5_000

# ── Model / training config ───────────────────────────────────────────────────
CONFIG = dict(
    emb_net         = [20, 15, 10],
    int_net         = [15, 10, 5],
    nb_steps        = 50,
    solver          = "CC",
    nb_flow         = 1,
    l1              = 0.5,
    b_size          = 128,
    nb_epoch        = 20,
    learning_rate   = 3e-4,
    nb_estop        = 20,
    nb_epoch_update = 50,
    n_mce_samples   = 2_000,
    mce_b_size      = 2_000,
)

RESULTS_DIR = Path(__file__).parent / "results" / "setting_piecewise_copula"


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(n_samples, rho_neg, rho_pos, seed):
    """
    Sample from the piecewise Gaussian copula DGP.

    Returns FloatTensor [n_samples, 2] with columns (X, Y).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    x = rng.uniform(-1, 1, n_samples).astype(np.float32)

    # Map X to its standard-normal score via the uniform CDF
    u_x = np.clip((x + 1) / 2, 1e-6, 1 - 1e-6)
    z_x = scipy_norm.ppf(u_x).astype(np.float32)

    # Piecewise rho
    rho_vals = np.where(x < 0, rho_neg, rho_pos).astype(np.float32)

    # Conditional: Z_Y | Z_X ~ N(rho * Z_X, sqrt(1 - rho^2))
    cond_mean = rho_vals * z_x
    cond_std  = np.sqrt(1.0 - rho_vals ** 2).astype(np.float32)
    y = (rng.standard_normal(n_samples).astype(np.float32) * cond_std + cond_mean)

    data = torch.from_numpy(np.stack([x, y], axis=1))
    return data


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(assumed_rho, seed):
    """
    Generate data from the piecewise DGP, train with assumed_rho, return ATE.

    The ATE is estimated as E[Y|do(X=X1)] − E[Y|do(X=X0)] via MC inversion.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data                 = generate_data(N_SAMPLES, RHO_NEG, RHO_POS, seed)
    data_train, data_val = split_data(data)
    data_mu, data_sigma  = compute_normalisation(data_train, data_val)

    Z_Sigma = get_cov_matrix(assumed_rho)

    model = build_rho_gnf(
        Z_Sigma         = Z_Sigma,
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

    # ATE contrast: do(X = X0) vs do(X = X1)
    return estimate_ate(
        model,
        Z_Sigma        = Z_Sigma,
        n_mce_samples  = CONFIG["n_mce_samples"],
        mce_b_size     = CONFIG["mce_b_size"],
        device         = device,
        treatment_dim  = 0,
        treatment_vals = (X0, X1),   # (−0.5, 0.5)
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(rho_values, all_estimates, save_path):
    """
    Two-panel ATE figure.

    Left  — ρ-curve: mean estimated ATE ± std vs assumed ρ, with true ATE line
            and vertical markers at the two regime rhos (RHO_NEG, RHO_POS).
    Right — RMSE vs assumed ρ, same markers.
    """
    mean_ests = all_estimates.mean(axis=1)
    std_ests  = all_estimates.std(axis=1)
    errors    = all_estimates - TRUE_ATE
    rmses     = np.sqrt((errors ** 2).mean(axis=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Piecewise Gaussian Copula — "
        f"ρ={RHO_NEG} (X<0) · ρ={RHO_POS} (X≥0)\n"
        f"n={N_SAMPLES:,}  ·  {N_SEEDS} seeds  ·  "
        f"True ATE = {TRUE_ATE:.4f}  "
        f"(E[Y|do(X={X1})]={TRUE_EY1:.4f} − E[Y|do(X={X0})]={TRUE_EY0:.4f})",
        fontsize=10,
    )

    # ── Left: ρ-curve ─────────────────────────────────────────────────────────
    ax1.plot(rho_values, mean_ests, "o-", color="steelblue",
             linewidth=2, markersize=5, label="Mean ATE estimate")
    ax1.fill_between(rho_values,
                     mean_ests - std_ests,
                     mean_ests + std_ests,
                     alpha=0.2, color="steelblue", label="±1 std (across seeds)")
    ax1.axhline(TRUE_ATE, color="red",    linestyle="--", linewidth=1.5,
                label=f"True ATE = {TRUE_ATE:.4f}")
    ax1.axvline(RHO_NEG,  color="purple", linestyle=":",  linewidth=1.5,
                label=f"ρ_neg = {RHO_NEG}  (X<0 regime)")
    ax1.axvline(RHO_POS,  color="green",  linestyle=":",  linewidth=1.5,
                label=f"ρ_pos = {RHO_POS}  (X≥0 regime)")

    ax1.set_xlabel("Assumed ρ (model's Gaussian copula)")
    ax1.set_ylabel("Estimated ATE")
    ax1.set_title("ρ-curve: mean ATE ± 1 std")
    ax1.legend(fontsize=8)

    # ── Right: RMSE vs assumed ρ ──────────────────────────────────────────────
    ax2.plot(rho_values, rmses, "s-", color="darkorange",
             linewidth=2, markersize=5, label="RMSE")
    ax2.axvline(RHO_NEG, color="purple", linestyle=":", linewidth=1.5,
                label=f"ρ_neg = {RHO_NEG}")
    ax2.axvline(RHO_POS, color="green",  linestyle=":", linewidth=1.5,
                label=f"ρ_pos = {RHO_POS}")

    best_idx = int(np.argmin(rmses))
    ax2.scatter(rho_values[best_idx], rmses[best_idx],
                color="red", zorder=5, s=80,
                label=f"Min RMSE at ρ = {rho_values[best_idx]:.2f}")

    ax2.set_xlabel("Assumed ρ (model's Gaussian copula)")
    ax2.set_ylabel("RMSE vs true ATE")
    ax2.set_title("RMSE vs Assumed ρ\n(lower = closer to true ATE)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Setting — piecewise Gaussian copula\n"
        f"  X<0 → ρ = {RHO_NEG},  X≥0 → ρ = {RHO_POS}\n"
        f"  ATE reference: do(X={X0}) vs do(X={X1})\n"
        f"  True EY0 = {TRUE_EY0:.5f},  True EY1 = {TRUE_EY1:.5f}\n"
        f"  True ATE = {TRUE_ATE:.5f}\n"
        f"  {len(RHO_VALUES)} assumed-ρ values × {N_SEEDS} seeds "
        f"= {len(RHO_VALUES) * N_SEEDS} runs\n"
    )

    all_estimates = np.zeros((len(RHO_VALUES), N_SEEDS))

    for i, rho in enumerate(RHO_VALUES):
        for s in range(N_SEEDS):
            print(f"[assumed_rho={rho:+.3f}  seed={s}]", end="  ", flush=True)
            ate = run_single(assumed_rho=rho, seed=s)
            all_estimates[i, s] = ate
            print(f"ATE_est={ate:+.4f}  err={ate - TRUE_ATE:+.4f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Piecewise Copula Summary  (true ATE = {TRUE_ATE:.4f}) ===")
    print(f"{'Assumed ρ':>10}  {'Mean est':>10}  {'Bias':>8}  {'RMSE':>8}")
    for i, rho in enumerate(RHO_VALUES):
        ests = all_estimates[i]
        bias = float(np.mean(ests - TRUE_ATE))
        rmse = float(np.sqrt(np.mean((ests - TRUE_ATE) ** 2)))
        print(f"{rho:>10.3f}  {np.mean(ests):>10.4f}  {bias:>8.4f}  {rmse:>8.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        RESULTS_DIR / "results.npz",
        rho_values    = RHO_VALUES,
        all_estimates = all_estimates,
        true_ate      = TRUE_ATE,
        true_ey0      = TRUE_EY0,
        true_ey1      = TRUE_EY1,
        rho_neg       = RHO_NEG,
        rho_pos       = RHO_POS,
        x0            = X0,
        x1            = X1,
    )
    print(f"Results saved → {RESULTS_DIR / 'results.npz'}")

    plot_results(RHO_VALUES, all_estimates, RESULTS_DIR / "ate_figure.png")


if __name__ == "__main__":
    main()
