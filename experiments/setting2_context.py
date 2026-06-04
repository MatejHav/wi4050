"""
Setting 2 — Heterogeneous Clayton confounding: V1 (no context) vs V2 (with context).

DGP:
    X  ~ Uniform(0, 1)                               observed confounder
    θ(X) = THETA_MIN + (THETA_MAX - THETA_MIN) * X  per-sample Clayton parameter
    (u_A, u_Y) ~ Clayton(θ(X_i))                    per-sample copula draw
    e_A = Φ⁻¹(u_A),  e_Y = Φ⁻¹(u_Y)               standard normal margins
    A  = e_A
    Y  = ALPHA * A + e_Y
    True ATE = ALPHA = 0.2

X is directly informative about confounding strength:
  X → 0  :  weak confounding  (θ → THETA_MIN, small tail dependence)
  X → 1  :  strong confounding (θ → THETA_MAX, large tail dependence)

Two model variants:
  V1 — no context : model sees only (A, Y); uses fixed assumed ρ
  V2 — context    : model sees (A, Y); X is passed as conditioning context
                    into the DAGConditioner embedding net (Part 0a)

The experiment sweeps over assumed ρ for both variants and compares:
  - ρ-curves (mean estimated ATE ± std)
  - RMSE vs assumed ρ

Reference line REF_RHO is the Gaussian-copula ρ that matches the mean Kendall's τ
across the X distribution (τ_mean ≈ 0.50 → REF_RHO ≈ 0.707).

Output
------
    results/setting2_context/results.npz      — all estimates for V1 and V2
    results/setting2_context/comparison.png   — side-by-side ρ-curve + RMSE plot
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
THETA_MIN  = 0.5
THETA_MAX  = 4.0

# Mean Kendall's τ over X ~ U(0,1):  E_X[θ(X)/(θ(X)+2)] ≈ 0.50
# REF_RHO = Gaussian-copula ρ with the same mean τ
TAU_MEAN   = 0.50
REF_RHO    = float(np.sin(np.pi * TAU_MEAN / 2))   # ≈ 0.707

# ── Sweep / experiment parameters ─────────────────────────────────────────────
RHO_VALUES = np.linspace(-0.99, 0.99, 10)
N_SEEDS    = 3
N_SAMPLES  = 5_000
CTX_DIM    = 1   # X is a scalar

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

RESULTS_DIR = Path(__file__).parent / "results" / "setting2_context"


# ── Data generation ───────────────────────────────────────────────────────────

def sample_clayton_copula_heterogeneous(n, theta_min, theta_max, rng):
    """
    Sample n pairs (u_A, u_Y) each from Clayton(θ_i) where θ_i = θ(X_i).
    Returns (u_A, u_Y, X) arrays, all length n.
    """
    X  = rng.uniform(0, 1, n)
    theta = theta_min + (theta_max - theta_min) * X

    u1 = rng.uniform(0, 1, n)
    v  = rng.uniform(0, 1, n)
    # Conditional CDF inversion for Clayton: u2 = (u1^{-θ}(v^{-θ/(θ+1)}-1)+1)^{-1/θ}
    inner = u1 ** (-theta) * (v ** (-theta / (theta + 1)) - 1) + 1
    u2    = np.clip(inner ** (-1.0 / theta), 1e-7, 1 - 1e-7)
    return u1, u2, X


def generate_data(n_samples, alpha, seed):
    """
    Returns:
        data     : FloatTensor [n, 2]  columns = (A, Y)
        context  : FloatTensor [n, 1]  column  = X
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    u_A, u_Y, X = sample_clayton_copula_heterogeneous(n_samples, THETA_MIN, THETA_MAX, rng)
    e_A = scipy_norm.ppf(u_A).astype(np.float32)
    e_Y = scipy_norm.ppf(u_Y).astype(np.float32)

    A = torch.from_numpy(e_A).unsqueeze(1)
    Y = alpha * A + torch.from_numpy(e_Y).unsqueeze(1)

    data    = torch.cat([A, Y], dim=1)
    context = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
    return data, context


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(assumed_rho, seed, use_context):
    """
    Train and evaluate one model.

    use_context=False → V1: context is discarded, cond_in=0
    use_context=True  → V2: context X is passed to the embedding net, cond_in=1
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data, context            = generate_data(N_SAMPLES, ALPHA, seed)
    data_train, data_val     = split_data(data)
    context_train, context_val = split_data(context)
    data_mu, data_sigma      = compute_normalisation(data_train, data_val)

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
        cond_in         = CTX_DIM if use_context else 0,
    )

    ctx_tr = context_train.to(device) if use_context else None
    ctx_va = context_val.to(device)   if use_context else None

    model, _ = train_model(
        model, data_train, data_val,
        nb_epoch      = CONFIG["nb_epoch"],
        b_size        = CONFIG["b_size"],
        nb_steps      = CONFIG["nb_steps"],
        learning_rate = CONFIG["learning_rate"],
        nb_estop      = CONFIG["nb_estop"],
        device        = device,
        context_train = ctx_tr,
        context_val   = ctx_va,
    )

    # For ATE estimation with V2, marginalise over observed X distribution
    ctx_mce = context.to(device) if use_context else None

    return estimate_ate(
        model,
        Z_Sigma         = Z_Sigma,
        n_mce_samples   = CONFIG["n_mce_samples"],
        mce_b_size      = CONFIG["mce_b_size"],
        device          = device,
        context_samples = ctx_mce,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(rho_values, est_v1, est_v2, save_path):
    """
    Two-panel figure comparing V1 (no context) and V2 (with context X).
    Left  — ρ-curves: mean ATE ± std for each variant
    Right — RMSE vs assumed ρ for each variant
    """
    def stats(est):
        mean = est.mean(axis=1)
        std  = est.std(axis=1)
        rmse = np.sqrt(((est - TRUE_ATE) ** 2).mean(axis=1))
        return mean, std, rmse

    mean1, std1, rmse1 = stats(est_v1)
    mean2, std2, rmse2 = stats(est_v2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: ρ-curves ────────────────────────────────────────────────────────
    for ax in (ax1,):
        ax.axhline(TRUE_ATE, color="red",   linestyle="--", linewidth=1.5,
                   label=f"True ATE = {TRUE_ATE}", zorder=1)
        ax.axvline(REF_RHO,  color="green", linestyle="--", linewidth=1.5,
                   label=f"REF ρ ≈ {REF_RHO:.3f}  (mean τ={TAU_MEAN:.2f})", zorder=1)

    ax1.plot(rho_values, mean1, "o-", color="darkorange",
             linewidth=2, markersize=5, label="V1 — no context")
    ax1.fill_between(rho_values, mean1 - std1, mean1 + std1,
                     alpha=0.2, color="darkorange")

    ax1.plot(rho_values, mean2, "s-", color="steelblue",
             linewidth=2, markersize=5, label="V2 — context X")
    ax1.fill_between(rho_values, mean2 - std2, mean2 + std2,
                     alpha=0.2, color="steelblue")

    ax1.set_xlabel("Assumed ρ (Gaussian model)")
    ax1.set_ylabel("Estimated ATE")
    ax1.set_title(
        f"ρ-curve  (heterogeneous Clayton θ∈[{THETA_MIN},{THETA_MAX}])\n"
        f"True ATE={TRUE_ATE}  ·  {N_SEEDS} seeds  ·  n={N_SAMPLES:,}"
    )
    ax1.legend(fontsize=8)

    # ── Right: RMSE ───────────────────────────────────────────────────────────
    ax2.axvline(REF_RHO, color="green", linestyle="--", linewidth=1.5,
                label=f"REF ρ ≈ {REF_RHO:.3f}", zorder=1)

    ax2.plot(rho_values, rmse1, "o-", color="darkorange",
             linewidth=2, markersize=5, label="V1 — no context")
    ax2.plot(rho_values, rmse2, "s-", color="steelblue",
             linewidth=2, markersize=5, label="V2 — context X")

    for vals, color, label in [(rmse1, "darkorange", "V1"), (rmse2, "steelblue", "V2")]:
        best_idx = int(np.argmin(vals))
        ax2.scatter(rho_values[best_idx], vals[best_idx],
                    color=color, zorder=5, s=80,
                    label=f"{label} min at ρ={rho_values[best_idx]:.2f}")

    ax2.set_xlabel("Assumed ρ (Gaussian model)")
    ax2.set_ylabel("RMSE")
    ax2.set_title(
        "RMSE vs Assumed ρ\n"
        "V1 (no context) vs V2 (context X)"
    )
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Setting 2 — heterogeneous Clayton θ∈[{THETA_MIN},{THETA_MAX}], "
        f"mean τ={TAU_MEAN:.2f}, REF_RHO={REF_RHO:.4f}\n"
        f"Variants: V1 (no context), V2 (context X)\n"
        f"ρ sweep: {len(RHO_VALUES)} values × {N_SEEDS} seeds × 2 variants "
        f"= {len(RHO_VALUES) * N_SEEDS * 2} runs\n"
    )

    est_v1 = np.zeros((len(RHO_VALUES), N_SEEDS))
    est_v2 = np.zeros((len(RHO_VALUES), N_SEEDS))

    for i, rho in enumerate(RHO_VALUES):
        for s in range(N_SEEDS):
            print(f"[ρ={rho:+.3f}  seed={s}]  V1", end="  ", flush=True)
            ate1 = run_single(assumed_rho=rho, seed=s, use_context=False)
            est_v1[i, s] = ate1
            print(f"ATE={ate1:+.4f}  |  V2", end="  ", flush=True)
            ate2 = run_single(assumed_rho=rho, seed=s, use_context=True)
            est_v2[i, s] = ate2
            print(f"ATE={ate2:+.4f}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Summary  (true ATE = {TRUE_ATE}) ===")
    print(f"{'ρ':>8}  {'V1 mean':>9}  {'V1 RMSE':>8}  {'V2 mean':>9}  {'V2 RMSE':>8}")
    for i, rho in enumerate(RHO_VALUES):
        v1_mean = np.mean(est_v1[i])
        v1_rmse = np.sqrt(np.mean((est_v1[i] - TRUE_ATE) ** 2))
        v2_mean = np.mean(est_v2[i])
        v2_rmse = np.sqrt(np.mean((est_v2[i] - TRUE_ATE) ** 2))
        marker  = " ←" if abs(rho - REF_RHO) == min(abs(RHO_VALUES - REF_RHO)) else ""
        print(f"{rho:>8.3f}  {v1_mean:>9.4f}  {v1_rmse:>8.4f}  {v2_mean:>9.4f}  {v2_rmse:>8.4f}{marker}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        RESULTS_DIR / "results.npz",
        rho_values = RHO_VALUES,
        est_v1     = est_v1,
        est_v2     = est_v2,
        true_ate   = TRUE_ATE,
        theta_min  = THETA_MIN,
        theta_max  = THETA_MAX,
        tau_mean   = TAU_MEAN,
        ref_rho    = REF_RHO,
    )
    print(f"Results saved → {RESULTS_DIR / 'results.npz'}")

    plot_results(RHO_VALUES, est_v1, est_v2, RESULTS_DIR / "comparison.png")


if __name__ == "__main__":
    main()
