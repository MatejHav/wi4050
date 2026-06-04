"""
Setting 0 & 1 — learned interventional Y distributions as assumed ρ varies.

One fixed dataset per setting (true ρ=0.3, α=1).  The rho-GNF is trained
seven times per setting, each time with a different *assumed* ρ.  After
training, the flow is inverted to obtain Y|do(A=0) and Y|do(A=1) samples.

Goal: show how the learned interventional distributions shift with assumed ρ,
while the observed Y (orange histogram) stays identical across columns.
The true interventional targets N(0,1) and N(1,1) are dashed reference lines.

DGP
---
  Setting 0: (e_A, e_Y) ~ MVN(0, [[1, 0.3], [0.3, 1]]);  A=e_A, Y=A+e_Y
  Setting 1: (u_A,u_Y) ~ Clayton(θ≈0.857, τ=0.3);  e·=Φ⁻¹(u·);  same SCM

Output
------
  results/setting01_interventional_sweep/interventional_sweep.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import norm as scipy_norm

from experiments.utils import (
    get_cov_matrix,
    split_data,
    compute_normalisation,
    build_rho_gnf,
    train_model,
    sample_interventional_y,
)

# ── DGP parameters (fixed) ────────────────────────────────────────────────────
TRUE_RHO   = 0.3
TRUE_ALPHA = 1.0
N_SAMPLES  = 5_000
SEED       = 0

# ── Assumed rho sweep ─────────────────────────────────────────────────────────
# Edge values ±1 are clipped to ±0.99 (MVN needs positive-definite Σ)
RHO_SWEEP_RAW = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
RHO_SWEEP     = [float(np.clip(r, -0.99, 0.99)) for r in RHO_SWEEP_RAW]

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
    n_mce_samples    = 3_000,
    mce_b_size       = 3_000,
)

RESULTS_DIR = Path(__file__).parent / "results" / "setting01_interventional_sweep"


# ── Clayton copula helpers ────────────────────────────────────────────────────

def rho_to_clayton_theta(rho: float) -> float:
    """Gaussian-copula Kendall's τ → Clayton θ."""
    tau = (2.0 / np.pi) * np.arcsin(np.clip(rho, -1 + 1e-9, 1 - 1e-9))
    if abs(tau) < 1e-9:
        return 0.0
    return float(np.clip(2.0 * tau / (1.0 - tau), -1.0 + 1e-6, 50.0))


def sample_clayton_copula(n: int, theta: float, rng) -> tuple:
    if theta >= 50.0:
        u1 = rng.uniform(0.0, 1.0, n); return u1, u1.copy()
    if theta <= -1.0 + 1e-5:
        u1 = rng.uniform(0.0, 1.0, n); return u1, 1.0 - u1
    if abs(theta) < 1e-6:
        return rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
    u1 = rng.uniform(0.0, 1.0, n)
    v  = rng.uniform(0.0, 1.0, n)
    inner = u1 ** (-theta) * (v ** (-theta / (theta + 1.0)) - 1.0) + 1.0
    u2 = np.clip(np.maximum(inner, 1e-300) ** (-1.0 / theta), 1e-7, 1 - 1e-7)
    return u1, u2


# ── Data generation ───────────────────────────────────────────────────────────

def generate_gaussian_data(n, alpha, rho, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Z_Sigma = get_cov_matrix(rho)
    noise = torch.distributions.MultivariateNormal(
        torch.zeros(2), Z_Sigma
    ).sample([n])
    A = noise[:, 0:1]
    Y = alpha * A + noise[:, 1:2]
    return torch.cat([A, Y], dim=1)


def generate_clayton_data(n, alpha, rho, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    theta = rho_to_clayton_theta(rho)
    u1, u2 = sample_clayton_copula(n, theta, rng)
    e_A = scipy_norm.ppf(u1).astype(np.float32)
    e_Y = scipy_norm.ppf(u2).astype(np.float32)
    A = torch.from_numpy(e_A).unsqueeze(1)
    Y = alpha * A + torch.from_numpy(e_Y).unsqueeze(1)
    return torch.cat([A, Y], dim=1)


# ── Single training + inversion run ──────────────────────────────────────────

def run_single(data, assumed_rho):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_train, data_val = split_data(data)
    data_mu, data_sigma  = compute_normalisation(data_train, data_val)
    Z_Sigma              = get_cov_matrix(assumed_rho)

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
    y_do0, y_do1 = sample_interventional_y(
        model,
        Z_Sigma       = Z_Sigma,
        n_mce_samples = CONFIG["n_mce_samples"],
        mce_b_size    = CONFIG["mce_b_size"],
        device        = device,
        treatment_vals= (0., 1.),
    )
    return y_do0, y_do1


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(Y_obs_g, Y_obs_c, results_g, results_c, save_path):
    """
    results_g / results_c : list of (y_do0, y_do1) tuples, one per assumed rho.
    """
    n_rho  = len(RHO_SWEEP)
    x_lo, x_hi = -5.5, 6.5
    x_ref = np.linspace(x_lo, x_hi, 500)

    # True interventional densities
    pdf_do0_true = scipy_norm.pdf(x_ref, 0.0,        1.0)
    pdf_do1_true = scipy_norm.pdf(x_ref, TRUE_ALPHA,  1.0)

    bins = np.linspace(x_lo, x_hi, 70)

    fig = plt.figure(figsize=(3.0 * n_rho, 7.0))
    gs  = gridspec.GridSpec(2, n_rho, figure=fig, hspace=0.60, wspace=0.25)

    row_info = [
        ("Setting 0 — Gaussian DGP", Y_obs_g, results_g),
        ("Setting 1 — Clayton DGP",  Y_obs_c, results_c),
    ]

    for row, (row_label, Y_obs, results) in enumerate(row_info):
        for col, (rho_raw, rho, (y_do0, y_do1)) in enumerate(
                zip(RHO_SWEEP_RAW, RHO_SWEEP, results)):

            ax = fig.add_subplot(gs[row, col])

            # ── Observed Y — fixed, same in every column ──────────────────────
            ax.hist(Y_obs, bins=bins, density=True, alpha=0.40,
                    color="tab:orange", label="Y observed" if col == 0 else None)

            # ── Learned interventional distributions ──────────────────────────
            ax.hist(y_do0, bins=bins, density=True, alpha=0.45,
                    color="tab:blue",
                    label="Y|do(A=0) learned" if col == 0 else None)
            ax.hist(y_do1, bins=bins, density=True, alpha=0.45,
                    color="tab:green",
                    label="Y|do(A=1) learned" if col == 0 else None)

            # ── True interventional targets (dashed) ──────────────────────────
            ax.plot(x_ref, pdf_do0_true, "b--", linewidth=1.2, alpha=0.8,
                    label="N(0,1) true" if col == 0 else None)
            ax.plot(x_ref, pdf_do1_true, "g--", linewidth=1.2, alpha=0.8,
                    label="N(1,1) true" if col == 0 else None)

            # ── Learned ATE annotation ────────────────────────────────────────
            learned_ate = float(np.mean(y_do1) - np.mean(y_do0))
            ax.set_title(
                f"assumed ρ = {rho_raw:+.2f}\n"
                f"learned ATE = {learned_ate:+.3f}",
                fontsize=8, pad=3,
            )

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=7)

            if col == 0:
                ax.set_ylabel(row_label, fontsize=8.5)
                ax.legend(fontsize=6, loc="upper left",
                          framealpha=0.8, handlelength=1.0)

    # True ATE reference line in title
    fig.suptitle(
        f"Learned Y|do(A=a) distributions as assumed ρ varies\n"
        f"True DGP: ρ={TRUE_RHO}, α={TRUE_ALPHA} (true ATE={TRUE_ALPHA:.0f}).  "
        f"Observed Y (orange) is identical across columns — only the model changes.",
        fontsize=10, y=1.02,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate one fixed dataset per setting
    print("Generating datasets ...")
    data_g = generate_gaussian_data(N_SAMPLES, TRUE_ALPHA, TRUE_RHO, seed=SEED)
    data_c = generate_clayton_data( N_SAMPLES, TRUE_ALPHA, TRUE_RHO, seed=SEED)

    Y_obs_g = data_g[:, 1].numpy()
    Y_obs_c = data_c[:, 1].numpy()

    results_g, results_c = [], []

    for rho_raw, rho in zip(RHO_SWEEP_RAW, RHO_SWEEP):
        print(f"\n=== assumed rho = {rho_raw:+.2f}  (clamped: {rho:+.3f}) ===")

        print("  [Setting 0 — Gaussian] training ...", flush=True)
        y_do0_g, y_do1_g = run_single(data_g, rho)
        ate_g = float(np.mean(y_do1_g) - np.mean(y_do0_g))
        print(f"  ATE_est = {ate_g:+.4f}  (true = {TRUE_ALPHA:+.1f})")
        results_g.append((y_do0_g, y_do1_g))

        print("  [Setting 1 — Clayton]   training ...", flush=True)
        y_do0_c, y_do1_c = run_single(data_c, rho)
        ate_c = float(np.mean(y_do1_c) - np.mean(y_do0_c))
        print(f"  ATE_est = {ate_c:+.4f}  (true = {TRUE_ALPHA:+.1f})")
        results_c.append((y_do0_c, y_do1_c))

    plot_results(Y_obs_g, Y_obs_c, results_g, results_c,
                 RESULTS_DIR / "interventional_sweep.png")


if __name__ == "__main__":
    main()
