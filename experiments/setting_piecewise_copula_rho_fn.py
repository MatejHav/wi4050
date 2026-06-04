"""
Setting: Piecewise Gaussian Copula with user-supplied rho_fn — ATE recovery demo.

DGP
---
    X       ~ Uniform(-1, 1)
    Z_X      = Φ⁻¹((X + 1) / 2)
    if X < 0:  Z_Y | Z_X ~ N(RHO_NEG · Z_X, √(1 − RHO_NEG²))   rho = −0.5
    if X ≥ 0:  Z_Y | Z_X ~ N(RHO_POS · Z_X, √(1 − RHO_POS²))   rho =  0.3
    Y  = Z_Y

True ATE (reference contrast x₀=−0.5 vs x₁=0.5)
-------------------------------------------------
    E[Y | do(X=x)] = ρ(x) · Φ⁻¹((x+1)/2)
    EY0 ≈  0.337,  EY1 ≈  0.202,  ATE ≈ −0.135

Experiment
----------
Train the rho-GNF with the TRUE rho_fn supplied as a Python callable:

    rho_fn(x) = −0.5  if x < 0
                 0.3  if x ≥ 0

Compare ATE recovery across N_SEEDS seeds against two fixed-rho baselines:
  • rho = 0  (misspecified, ignores all confounding correlation)
  • rho = −0.1  (average of the two regimes — 50/50 split)

Output
------
    results/setting_piecewise_copula_rho_fn/results.npz
    results/setting_piecewise_copula_rho_fn/ate_figure.png
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
    build_rho_fn_gnf,
    train_model,
    estimate_ate,
)

# ── DGP parameters ────────────────────────────────────────────────────────────
RHO_NEG = -0.5
RHO_POS =  0.3
RHO_AVG = (RHO_NEG + RHO_POS) / 2.0   # −0.1  (50/50 mixture mean)

X0 = -0.5
X1 =  0.5

_u0 = np.clip((X0 + 1) / 2, 1e-6, 1 - 1e-6)
_u1 = np.clip((X1 + 1) / 2, 1e-6, 1 - 1e-6)
TRUE_EY0 = float(RHO_NEG * scipy_norm.ppf(_u0))
TRUE_EY1 = float(RHO_POS * scipy_norm.ppf(_u1))
TRUE_ATE  = TRUE_EY1 - TRUE_EY0

# ── Experiment parameters ─────────────────────────────────────────────────────
N_SEEDS   = 5
N_SAMPLES = 1_000

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

RESULTS_DIR = Path(__file__).parent / "results" / "setting_piecewise_copula_rho_fn"


# ── rho function ──────────────────────────────────────────────────────────────

def rho_fn(x: torch.Tensor) -> torch.Tensor:
    """True piecewise rho: −0.5 for X < 0, 0.3 for X ≥ 0. Input/output [B, 1]."""
    return torch.where(x < 0, torch.full_like(x, RHO_NEG), torch.full_like(x, RHO_POS))


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(n_samples, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    x = rng.uniform(-1, 1, n_samples).astype(np.float32)
    u_x  = np.clip((x + 1) / 2, 1e-6, 1 - 1e-6)
    z_x  = scipy_norm.ppf(u_x).astype(np.float32)
    rhos = np.where(x < 0, RHO_NEG, RHO_POS).astype(np.float32)
    cond_mean = rhos * z_x
    cond_std  = np.sqrt(1.0 - rhos ** 2).astype(np.float32)
    y = rng.standard_normal(n_samples).astype(np.float32) * cond_std + cond_mean

    return torch.from_numpy(np.stack([x, y], axis=1))


# ── Single runs ───────────────────────────────────────────────────────────────

def run_rho_fn(seed):
    """Train with the true rho_fn and return ATE estimate."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data                 = generate_data(N_SAMPLES, seed)
    data_train, data_val = split_data(data)
    data_mu, data_sigma  = compute_normalisation(data_train, data_val)

    model = build_rho_fn_gnf(
        rho_fn          = rho_fn,
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
        rho_x_col     = 0,   # X is column 0 — pass it to rho_fn each batch
    )

    # Z marginals are N(0,1) by the Gaussian copula property, so identity is correct.
    Z_Sigma_id = get_cov_matrix(0.0)
    return estimate_ate(
        model,
        Z_Sigma        = Z_Sigma_id,
        n_mce_samples  = CONFIG["n_mce_samples"],
        mce_b_size     = CONFIG["mce_b_size"],
        device         = device,
        treatment_dim  = 0,
        treatment_vals = (X0, X1),
    )


def run_fixed_rho(assumed_rho, seed):
    """Train with a fixed rho and return ATE estimate."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data                 = generate_data(N_SAMPLES, seed)
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
    return estimate_ate(
        model,
        Z_Sigma        = Z_Sigma,
        n_mce_samples  = CONFIG["n_mce_samples"],
        mce_b_size     = CONFIG["mce_b_size"],
        device         = device,
        treatment_dim  = 0,
        treatment_vals = (X0, X1),
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(est_rho_fn, est_rho0, est_avg, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Piecewise Copula — rho_fn vs fixed-ρ baselines\n"
        f"DGP: ρ={RHO_NEG} (X<0) · ρ={RHO_POS} (X≥0) · "
        f"True ATE = {TRUE_ATE:.4f}  (n={N_SAMPLES}, {N_SEEDS} seeds)",
        fontsize=10,
    )

    variants = [
        (est_rho_fn, "rho_fn (true)",  "steelblue",  "o"),
        (est_rho0,   "fixed ρ=0",      "darkorange",  "s"),
        (est_avg,    f"fixed ρ={RHO_AVG:.1f} (avg)", "purple", "^"),
    ]

    jitter = np.linspace(-0.15, 0.15, N_SEEDS)
    for k, (ests, label, color, marker) in enumerate(variants):
        xpos = k + 1
        ax.scatter(xpos + jitter, ests, color=color, marker=marker,
                   alpha=0.7, zorder=3, s=60, label=f"{label}  (mean={np.mean(ests):.4f})")
        ax.errorbar(xpos, np.mean(ests), yerr=np.std(ests),
                    fmt="none", color=color, capsize=6, linewidth=2, zorder=4)

    ax.axhline(TRUE_ATE, color="red", linestyle="--", linewidth=1.5,
               label=f"True ATE = {TRUE_ATE:.4f}", zorder=2)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["rho_fn (true)", "fixed ρ=0", f"fixed ρ={RHO_AVG:.1f}"])
    ax.set_ylabel("Estimated ATE")
    ax.set_title("ATE estimates (dots = individual seeds, bar = ±1 std)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Setting — piecewise copula with rho_fn\n"
        f"  rho_fn(x) = {RHO_NEG} (x<0)  |  {RHO_POS} (x>=0)\n"
        f"  True EY0 = {TRUE_EY0:.5f},  True EY1 = {TRUE_EY1:.5f}\n"
        f"  True ATE = {TRUE_ATE:.5f}\n"
        f"  {N_SEEDS} seeds, n={N_SAMPLES}\n"
    )

    est_rho_fn = np.zeros(N_SEEDS)
    est_rho0   = np.zeros(N_SEEDS)
    est_avg    = np.zeros(N_SEEDS)

    for s in range(N_SEEDS):
        print(f"[seed={s}]  rho_fn", end="  ", flush=True)
        est_rho_fn[s] = run_rho_fn(seed=s)
        print(f"ATE={est_rho_fn[s]:+.4f}  |  fixed rho=0", end="  ", flush=True)
        est_rho0[s] = run_fixed_rho(assumed_rho=0.0, seed=s)
        print(f"ATE={est_rho0[s]:+.4f}  |  fixed rho={RHO_AVG:.1f}", end="  ", flush=True)
        est_avg[s] = run_fixed_rho(assumed_rho=RHO_AVG, seed=s)
        print(f"ATE={est_avg[s]:+.4f}", flush=True)

    print(f"\n=== Summary  (true ATE = {TRUE_ATE:.4f}) ===")
    header = f"{'Variant':<22}  {'Mean':>8}  {'Bias':>8}  {'RMSE':>8}"
    print(header)
    for label, ests in [
        ("rho_fn (true)",       est_rho_fn),
        ("fixed rho=0",         est_rho0),
        (f"fixed rho={RHO_AVG:.1f}", est_avg),
    ]:
        mean = float(np.mean(ests))
        bias = float(np.mean(ests - TRUE_ATE))
        rmse = float(np.sqrt(np.mean((ests - TRUE_ATE) ** 2)))
        print(f"{label:<22}  {mean:>8.4f}  {bias:>8.4f}  {rmse:>8.4f}")

    np.savez(
        RESULTS_DIR / "results.npz",
        est_rho_fn = est_rho_fn,
        est_rho0   = est_rho0,
        est_avg    = est_avg,
        true_ate   = TRUE_ATE,
        true_ey0   = TRUE_EY0,
        true_ey1   = TRUE_EY1,
        rho_neg    = RHO_NEG,
        rho_pos    = RHO_POS,
        rho_avg    = RHO_AVG,
        x0         = X0,
        x1         = X1,
    )
    print(f"Results saved → {RESULTS_DIR / 'results.npz'}")

    plot_results(est_rho_fn, est_rho0, est_avg, RESULTS_DIR / "ate_figure.png")


if __name__ == "__main__":
    main()
