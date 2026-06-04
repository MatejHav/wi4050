"""
Setting 0 — Baseline: Gaussian copula assumption satisfied.

DGP
---
    (e_A, e_Y) ~ MVN(0, [[1, rho], [rho, 1]])
    A = e_A
    Y = alpha * A + e_Y

The model is given the exact true rho, so the Gaussian copula assumption holds perfectly.
This is the control condition. We expect near-zero bias with low variance.

Output
------
    results/setting0/ate_estimates.npy   — raw ATE estimates, shape [K]
    results/setting0/errors.npy          — ATE_estimated - TRUE_ATE, shape [K]
    results/setting0/plot1_error_distribution.png
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
RHO      = -0.55   # true rho; given exactly to the model

# ── Experiment parameters ─────────────────────────────────────────────────────
K        = 100
N_SAMPLES = 5_000

# ── Model / training config ───────────────────────────────────────────────────
CONFIG = dict(
    emb_net          = [20, 15, 10],   # last entry = embedding size
    int_net          = [15, 10, 5],    # UMNN integrand hidden layers
    nb_steps         = 50,             # integration steps
    solver           = "CC",
    nb_flow          = 1,
    l1               = 0.5,
    b_size           = 128,
    nb_epoch         = 3_000,
    learning_rate    = 3e-4,
    nb_estop         = 100,            # early stopping patience (epochs)
    nb_epoch_update  = 50,             # how often to update DAG dual params
    n_mce_samples    = 2_000,
    mce_b_size       = 2_000,
)

RESULTS_DIR = Path(__file__).parent / "results" / "setting0"


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(n_samples, alpha, rho, seed):
    """
    Hoover model with Gaussian copula and unit-variance marginals.
    The model's assumed Z_Sigma = [[1, rho], [rho, 1]] matches the DGP exactly.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Z_Sigma = get_cov_matrix(rho)
    noise = torch.distributions.MultivariateNormal(
        torch.zeros(2), Z_Sigma
    ).sample([n_samples])

    A = noise[:, 0:1]
    Y = alpha * A + noise[:, 1:2]
    return torch.cat([A, Y], dim=1)  # [N, 2]


# ── Single experiment ─────────────────────────────────────────────────────────

def run_single(seed):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data                   = generate_data(N_SAMPLES, ALPHA, RHO, seed)
    data_train, data_val   = split_data(data)
    data_mu, data_sigma    = compute_normalisation(data_train, data_val)
    Z_Sigma                = get_cov_matrix(RHO)

    model = build_rho_gnf(
        Z_Sigma          = Z_Sigma,
        emb_net          = CONFIG["emb_net"],
        int_net          = CONFIG["int_net"],
        nb_steps         = CONFIG["nb_steps"],
        solver           = CONFIG["solver"],
        l1               = CONFIG["l1"],
        nb_flow          = CONFIG["nb_flow"],
        data_mu          = data_mu,
        data_sigma       = data_sigma,
        nb_epoch_update  = CONFIG["nb_epoch_update"],
        device           = device,
    )

    model, _ = train_model(
        model,
        data_train,
        data_val,
        nb_epoch      = CONFIG["nb_epoch"],
        b_size        = CONFIG["b_size"],
        nb_steps      = CONFIG["nb_steps"],
        learning_rate = CONFIG["learning_rate"],
        nb_estop      = CONFIG["nb_estop"],
        device        = device,
    )

    ate = estimate_ate(
        model,
        Z_Sigma       = Z_Sigma,
        n_mce_samples = CONFIG["n_mce_samples"],
        mce_b_size    = CONFIG["mce_b_size"],
        device        = device,
    )
    return ate


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(errors, save_path):
    bias = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=20, edgecolor="black", alpha=0.75, color="steelblue",
            label="Error distribution")
    ax.axvline(0,    color="red",    linestyle="--", linewidth=1.5, label="Zero bias")
    ax.axvline(bias, color="orange", linestyle="-",  linewidth=1.5,
               label=f"Mean bias = {bias:.4f}")

    ax.set_xlabel("ATE_estimated − ATE_true")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Setting 0 — Baseline (Gaussian copula, ρ = {RHO})\n"
        f"K = {len(errors)} runs,  n = {N_SAMPLES:,},  "
        f"Bias = {bias:.4f},  RMSE = {rmse:.4f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ate_estimates = []
    for k in range(K):
        print(f"[{k + 1:>3}/{K}] seed={k}", end="  ", flush=True)
        ate = run_single(seed=k)
        ate_estimates.append(ate)
        print(f"ATE_est = {ate:.4f}  (true = {TRUE_ATE:.4f})", flush=True)

    ate_estimates = np.array(ate_estimates)
    errors = ate_estimates - TRUE_ATE

    np.save(RESULTS_DIR / "ate_estimates.npy", ate_estimates)
    np.save(RESULTS_DIR / "errors.npy", errors)

    print(f"\n=== Setting 0 — Summary ===")
    print(f"  Mean bias  : {errors.mean():.4f}")
    print(f"  Std        : {errors.std():.4f}")
    print(f"  RMSE       : {np.sqrt((errors ** 2).mean()):.4f}")
    print(f"  Min / Max  : {errors.min():.4f} / {errors.max():.4f}")

    plot_results(errors, RESULTS_DIR / "plot1_error_distribution.png")


if __name__ == "__main__":
    main()
