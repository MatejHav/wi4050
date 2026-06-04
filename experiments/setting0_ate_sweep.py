"""
Setting 0 — ATE sweep: RMSE vs true ATE magnitude.

For each alpha value in ALPHA_VALUES, trains the rho-GNF over N_SEEDS independent
datasets and computes the RMSE of the estimated ATE against the true ATE.

DGP (same as Setting 0 baseline):
    (e_A, e_Y) ~ MVN(0, [[1, rho], [rho, 1]])
    A = e_A
    Y = alpha * A + e_Y

The model is given the exact true rho throughout.

Output
------
    results/setting0_ate_sweep/results.npz         — all estimates and errors
    results/setting0_ate_sweep/rmse_vs_ate.png     — RMSE per alpha with scatter
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

# ── Sweep parameters ──────────────────────────────────────────────────────────
ALPHA_VALUES = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
RHO          = -0.55
N_SEEDS      = 5
N_SAMPLES    = 5_000

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
    nb_estop         = 20,        # no early stopping within 20 epochs
    nb_epoch_update  = 50,        # dual param update never triggers in 20 epochs
    n_mce_samples    = 2_000,
    mce_b_size       = 2_000,
)

RESULTS_DIR = Path(__file__).parent / "results" / "setting0_ate_sweep"


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(n_samples, alpha, rho, seed):
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

def run_single(alpha, seed):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data                 = generate_data(N_SAMPLES, alpha, RHO, seed)
    data_train, data_val = split_data(data)
    data_mu, data_sigma  = compute_normalisation(data_train, data_val)
    Z_Sigma              = get_cov_matrix(RHO)

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
    return estimate_ate(
        model,
        Z_Sigma       = Z_Sigma,
        n_mce_samples = CONFIG["n_mce_samples"],
        mce_b_size    = CONFIG["mce_b_size"],
        device        = device,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(alpha_values, all_estimates, save_path):
    """
    Two-panel figure:
      Left  — RMSE per alpha with ±1 std band
      Right — scatter of individual estimates vs true ATE (diagonal = perfect)
    """
    alpha_arr = np.array(alpha_values)
    rmses, biases, stds = [], [], []

    for i, alpha in enumerate(alpha_values):
        ests   = all_estimates[i]          # shape [N_SEEDS]
        errors = ests - alpha
        rmses.append(np.sqrt(np.mean(errors ** 2)))
        biases.append(np.mean(errors))
        stds.append(np.std(errors))

    rmses  = np.array(rmses)
    biases = np.array(biases)
    stds   = np.array(stds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: RMSE vs true ATE ────────────────────────────────────────────────
    ax1.plot(alpha_arr, rmses, "o-", color="steelblue", linewidth=2,
             markersize=7, label="RMSE")
    ax1.fill_between(alpha_arr, rmses - stds, rmses + stds,
                     alpha=0.2, color="steelblue", label="±1 std of error")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("True ATE (α)")
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE vs True ATE\n"
                  f"(Setting 0, {N_SEEDS} seeds, {CONFIG['nb_epoch']} epochs, "
                  f"n={N_SAMPLES:,})")
    ax1.legend()

    # ── Right: estimated vs true ATE scatter ──────────────────────────────────
    for i, alpha in enumerate(alpha_values):
        ests = all_estimates[i]
        ax2.scatter([alpha] * N_SEEDS, ests, alpha=0.7, s=40,
                    label=f"α={alpha:.1f}" if i == 0 else "_nolegend_")

    lo = min(alpha_values) - 0.3
    hi = max(alpha_values) + 0.3
    ax2.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect calibration")
    ax2.set_xlabel("True ATE (α)")
    ax2.set_ylabel("ATE_estimated")
    ax2.set_title("Estimated vs True ATE\n(each dot = one seed)")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # all_estimates[i, s] = ATE estimate for alpha_values[i], seed s
    all_estimates = np.zeros((len(ALPHA_VALUES), N_SEEDS))

    for i, alpha in enumerate(ALPHA_VALUES):
        for s in range(N_SEEDS):
            print(f"[alpha={alpha:+.2f}  seed={s}]", end="  ", flush=True)
            ate = run_single(alpha=alpha, seed=s)
            all_estimates[i, s] = ate
            print(f"ATE_est={ate:+.4f}  (true={alpha:+.4f}  "
                  f"err={ate - alpha:+.4f})", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== ATE Sweep Summary ===")
    print(f"{'Alpha':>8}  {'Mean est':>10}  {'Bias':>8}  {'RMSE':>8}")
    for i, alpha in enumerate(ALPHA_VALUES):
        ests  = all_estimates[i]
        bias  = np.mean(ests - alpha)
        rmse  = np.sqrt(np.mean((ests - alpha) ** 2))
        print(f"{alpha:>8.3f}  {np.mean(ests):>10.4f}  {bias:>8.4f}  {rmse:>8.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        RESULTS_DIR / "results.npz",
        alpha_values  = np.array(ALPHA_VALUES),
        all_estimates = all_estimates,
    )

    plot_results(ALPHA_VALUES, all_estimates, RESULTS_DIR / "rmse_vs_ate.png")


if __name__ == "__main__":
    main()
