"""
Setting 0 — ρ sweep: RMSE vs assumed ρ when true ρ is fixed.

A dataset is generated with TRUE_RHO = 0.3 (the actual confounding).
The model is then trained with a range of *assumed* ρ values spanning [-1, +1],
while the data and true ATE stay fixed.

This is the ρ-curve experiment: it shows how sensitive the ATE estimate is to
misspecification of ρ, and at which assumed ρ the model recovers the true ATE.

DGP:
    (e_A, e_Y) ~ MVN(0, [[1, TRUE_RHO], [TRUE_RHO, 1]])
    A = e_A
    Y = ALPHA * A + e_Y
    True ATE = ALPHA = 0.2  (unaffected by rho — it is purely causal)

For each assumed ρ, N_SEEDS independent datasets (all with TRUE_RHO) are
generated and trained, giving a distribution of ATE estimates whose RMSE
against the true ATE is plotted as a function of the assumed ρ.

Output
------
    results/setting0_rho_sweep/results.npz        — all estimates and errors
    results/setting0_rho_sweep/rho_curve.png      — RMSE + ρ-curve plot
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.utils import (
    get_cov_matrix,
    split_data,
    compute_normalisation,
    build_rho_gnf,
    train_model,
    estimate_ate,
)

# ── DGP parameters ────────────────────────────────────────────────────────────
TRUE_ATE = 0.2
ALPHA    = 0.2
TRUE_RHO = 0.3   # the rho that actually generated the data

# ── Sweep parameters ──────────────────────────────────────────────────────────
# 20 assumed rho values from -1 to +1, excluding exact boundaries for
# numerical stability (MVN requires positive-definite Z_Sigma)
RHO_VALUES = np.linspace(-0.99, 0.99, 20)
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

RESULTS_DIR = Path(__file__).parent / "results" / "setting0_rho_sweep"


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(n_samples, alpha, rho, seed):
    """Always uses TRUE_RHO regardless of what assumed rho we pass to the model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Z_Sigma = get_cov_matrix(rho)
    noise = torch.distributions.MultivariateNormal(
        torch.zeros(2), Z_Sigma
    ).sample([n_samples])
    A = noise[:, 0:1]
    Y = alpha * A + noise[:, 1:2]
    return torch.cat([A, Y], dim=1)


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(assumed_rho, seed):
    """
    Generate data with TRUE_RHO, train with assumed_rho, return ATE estimate.
    The mismatch between TRUE_RHO and assumed_rho is the misspecification under test.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Data always comes from the TRUE_RHO DGP
    data                 = generate_data(N_SAMPLES, ALPHA, TRUE_RHO, seed)
    data_train, data_val = split_data(data)
    data_mu, data_sigma  = compute_normalisation(data_train, data_val)

    # Model is built with the *assumed* rho
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
    mean_ests = all_estimates.mean(axis=1)   # [n_rhos]
    std_ests  = all_estimates.std(axis=1)
    errors    = all_estimates - TRUE_ATE     # [n_rhos, N_SEEDS]
    rmses     = np.sqrt((errors ** 2).mean(axis=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: ρ-curve ─────────────────────────────────────────────────────────
    ax1.plot(rho_values, mean_ests, "o-", color="steelblue",
             linewidth=2, markersize=5, label="Mean ATE estimate")
    ax1.fill_between(rho_values,
                     mean_ests - std_ests,
                     mean_ests + std_ests,
                     alpha=0.2, color="steelblue", label="±1 std (across seeds)")
    ax1.axhline(TRUE_ATE,  color="red",    linestyle="--", linewidth=1.5,
                label=f"True ATE = {TRUE_ATE}")
    ax1.axvline(TRUE_RHO,  color="green",  linestyle="--", linewidth=1.5,
                label=f"True ρ = {TRUE_RHO}")

    ax1.set_xlabel("Assumed ρ")
    ax1.set_ylabel("Estimated ATE")
    ax1.set_title(
        f"ρ-curve  (true ρ = {TRUE_RHO}, true ATE = {TRUE_ATE})\n"
        f"{N_SEEDS} seeds · {CONFIG['nb_epoch']} epochs · n = {N_SAMPLES:,}"
    )
    ax1.legend(fontsize=8)

    # ── Right: RMSE vs assumed ρ ──────────────────────────────────────────────
    ax2.plot(rho_values, rmses, "s-", color="darkorange",
             linewidth=2, markersize=5, label="RMSE")
    ax2.axvline(TRUE_RHO, color="green", linestyle="--", linewidth=1.5,
                label=f"True ρ = {TRUE_RHO}")

    # Mark the assumed rho that gives minimum RMSE
    best_idx = int(np.argmin(rmses))
    ax2.scatter(rho_values[best_idx], rmses[best_idx],
                color="red", zorder=5, s=80,
                label=f"Min RMSE at ρ = {rho_values[best_idx]:.2f}")

    ax2.set_xlabel("Assumed ρ")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE vs Assumed ρ\n(lower = closer to true ATE)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # all_estimates[i, s] = ATE estimate for RHO_VALUES[i], seed s
    all_estimates = np.zeros((len(RHO_VALUES), N_SEEDS))

    for i, rho in enumerate(RHO_VALUES):
        for s in range(N_SEEDS):
            print(f"[assumed_rho={rho:+.3f}  seed={s}]", end="  ", flush=True)
            ate = run_single(assumed_rho=rho, seed=s)
            all_estimates[i, s] = ate
            print(f"ATE_est={ate:+.4f}  err={ate - TRUE_ATE:+.4f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== ρ Sweep Summary  (true ρ = {TRUE_RHO}, true ATE = {TRUE_ATE}) ===")
    print(f"{'Assumed ρ':>10}  {'Mean est':>10}  {'Bias':>8}  {'RMSE':>8}")
    for i, rho in enumerate(RHO_VALUES):
        ests = all_estimates[i]
        bias = np.mean(ests - TRUE_ATE)
        rmse = np.sqrt(np.mean((ests - TRUE_ATE) ** 2))
        marker = " ← true ρ" if abs(rho - TRUE_RHO) == min(abs(RHO_VALUES - TRUE_RHO)) else ""
        print(f"{rho:>10.3f}  {np.mean(ests):>10.4f}  {bias:>8.4f}  {rmse:>8.4f}{marker}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        RESULTS_DIR / "results.npz",
        rho_values    = RHO_VALUES,
        all_estimates = all_estimates,
        true_rho      = TRUE_RHO,
        true_ate      = TRUE_ATE,
    )

    plot_results(RHO_VALUES, all_estimates, RESULTS_DIR / "rho_curve.png")


if __name__ == "__main__":
    main()
