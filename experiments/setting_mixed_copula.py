"""
Setting: Regime-switching copula — Clayton (X<0) vs Gumbel (X≥0).

DGP
---
    X       ~ Uniform(-1, 1)           observed confounder
    if X < 0:  (u_A, u_Y) ~ Clayton(THETA_CLAYTON)   lower-tail dependence
    if X >= 0: (u_A, u_Y) ~ Gumbel(THETA_GUMBEL)    upper-tail dependence
    e_A = Φ⁻¹(u_A),  e_Y = Φ⁻¹(u_Y)               standard normal margins
    A   = e_A
    Y   = ALPHA * A + e_Y
    True ATE = ALPHA = 0.2

Both copulas are tuned to the same Kendall's τ = 0.5
(Clayton θ=2 → τ=0.5, Gumbel θ=2 → τ=0.5),
equivalent Gaussian ρ_ref ≈ 0.707.

X signals which copula family is active.  The Gaussian assumption is violated
in both regimes but in structurally opposite ways: Clayton creates lower-tail
co-movement, Gumbel creates upper-tail co-movement.

Two model variants
------------------
  V1 — no context: sees only (A,Y); uses a single fixed assumed ρ
  V2 — context X:  X is passed to the DAGConditioner embedding net (Part 0a)

Sweep
-----
  Assumed ρ ∈ [−0.99, 0.99]  (10 values × N_SEEDS seeds × 2 variants)
  Metric: estimated ATE via Monte Carlo flow inversion, RMSE vs true ATE.

Output
------
    results/setting_mixed_copula/results.npz
    results/setting_mixed_copula/comparison.png
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
TRUE_ATE      = 0.2
ALPHA         = 0.2
THETA_CLAYTON = 2.0          # Clayton θ → Kendall's τ = θ/(θ+2) = 0.5
THETA_GUMBEL  = 2.0          # Gumbel  θ → Kendall's τ = 1 − 1/θ  = 0.5
TAU_MEAN      = 0.5          # both regimes share the same marginal τ
REF_RHO       = float(np.sin(np.pi * TAU_MEAN / 2))   # ≈ 0.707

# ── Sweep / experiment parameters ─────────────────────────────────────────────
RHO_VALUES = np.linspace(-0.99, 0.99, 10)
N_SEEDS    = 3
N_SAMPLES  = 5_000
CTX_DIM    = 1

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

RESULTS_DIR = Path(__file__).parent / "results" / "setting_mixed_copula"


# ── Copula samplers ───────────────────────────────────────────────────────────

def _sample_clayton(n, theta, rng):
    """Sample n pairs (u1, u2) from Clayton(theta) via conditional CDF inversion."""
    u1 = rng.uniform(0, 1, n)
    v  = rng.uniform(0, 1, n)
    # C(u2|u1) = v  →  u2 = (u1^{-θ}(v^{-θ/(θ+1)}-1)+1)^{-1/θ}
    inner = u1 ** (-theta) * (v ** (-theta / (theta + 1)) - 1) + 1
    u2 = np.clip(inner ** (-1.0 / theta), 1e-7, 1 - 1e-7)
    return np.clip(u1, 1e-7, 1 - 1e-7), u2


def _sample_gumbel(n, theta, rng):
    """
    Sample n pairs (u1, u2) from Gumbel(theta) copula, theta >= 1.

    Uses conditional CDF inversion via bisection (50 iterations ≈ 1e-15 precision).

    The Gumbel conditional CDF:
        C(u2|u1) = C(u1,u2)/u1 * (-ln u1)^{θ-1} * ((-ln u1)^θ + (-ln u2)^θ)^{1/θ-1}
    is monotone increasing in u2, so bisection on [ε, 1-ε] always converges.
    """
    u1 = rng.uniform(1e-6, 1 - 1e-6, n)
    v  = rng.uniform(1e-6, 1 - 1e-6, n)   # target conditional value

    x        = (-np.log(u1)) ** theta
    x_tm1    = (-np.log(u1)) ** (theta - 1)  # (-ln u1)^{θ-1}

    def cond_minus_v(u2_arr):
        u2c  = np.clip(u2_arr, 1e-10, 1 - 1e-10)
        y    = (-np.log(u2c)) ** theta
        xpy  = x + y
        C    = np.exp(-xpy ** (1.0 / theta))
        cond = C / u1 * x_tm1 * xpy ** (1.0 / theta - 1)
        return cond - v

    lo = np.full(n, 1e-7)
    hi = np.full(n, 1 - 1e-7)
    for _ in range(50):
        mid = (lo + hi) * 0.5
        lo  = np.where(cond_minus_v(mid) < 0, mid, lo)
        hi  = np.where(cond_minus_v(mid) < 0, hi,  mid)

    u2 = (lo + hi) * 0.5
    return np.clip(u1, 1e-7, 1 - 1e-7), np.clip(u2, 1e-7, 1 - 1e-7)


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(n_samples, alpha, seed):
    """
    Sample from the regime-switching DGP.

    Returns
    -------
    data    : FloatTensor [n, 2]  columns = (A, Y)
    context : FloatTensor [n, 1]  column  = X
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X = rng.uniform(-1, 1, n_samples)               # observed confounder

    neg_mask = X < 0
    pos_mask = ~neg_mask
    n_neg    = int(neg_mask.sum())
    n_pos    = n_samples - n_neg

    u_A = np.empty(n_samples)
    u_Y = np.empty(n_samples)

    if n_neg > 0:
        u_A[neg_mask], u_Y[neg_mask] = _sample_clayton(n_neg, THETA_CLAYTON, rng)
    if n_pos > 0:
        u_A[pos_mask], u_Y[pos_mask] = _sample_gumbel(n_pos, THETA_GUMBEL, rng)

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
    Train one model and return its ATE estimate.

    use_context=False  →  V1: context X is discarded
    use_context=True   →  V2: context X conditions the DAGConditioner
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data, context           = generate_data(N_SAMPLES, ALPHA, seed)
    data_train, data_val    = split_data(data)
    ctx_train,  ctx_val     = split_data(context)
    data_mu, data_sigma     = compute_normalisation(data_train, data_val)

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

    model, _ = train_model(
        model, data_train, data_val,
        nb_epoch      = CONFIG["nb_epoch"],
        b_size        = CONFIG["b_size"],
        nb_steps      = CONFIG["nb_steps"],
        learning_rate = CONFIG["learning_rate"],
        nb_estop      = CONFIG["nb_estop"],
        device        = device,
        context_train = ctx_train if use_context else None,
        context_val   = ctx_val   if use_context else None,
    )

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
    Two-panel figure: ρ-curve (left) and RMSE vs ρ (right).
    """
    def stats(est):
        mean = est.mean(axis=1)
        std  = est.std(axis=1)
        rmse = np.sqrt(((est - TRUE_ATE) ** 2).mean(axis=1))
        return mean, std, rmse

    mean1, std1, rmse1 = stats(est_v1)
    mean2, std2, rmse2 = stats(est_v2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Regime-switching copula: Clayton (X<0, θ={THETA_CLAYTON}) "
        f"vs Gumbel (X≥0, θ={THETA_GUMBEL})\n"
        f"n={N_SAMPLES:,}  ·  {N_SEEDS} seeds  ·  True ATE={TRUE_ATE}",
        fontsize=11,
    )

    # ── Left: ρ-curves ────────────────────────────────────────────────────────
    ax1.axhline(TRUE_ATE, color="red",   linestyle="--", linewidth=1.5,
                label=f"True ATE = {TRUE_ATE}", zorder=1)
    ax1.axvline(REF_RHO,  color="green", linestyle="--", linewidth=1.5,
                label=f"ρ_ref ≈ {REF_RHO:.3f}  (τ={TAU_MEAN:.1f}, matched)", zorder=1)

    ax1.plot(rho_values, mean1, "o-", color="darkorange",
             linewidth=2, markersize=5, label="V1 — no context")
    ax1.fill_between(rho_values, mean1 - std1, mean1 + std1,
                     alpha=0.2, color="darkorange")

    ax1.plot(rho_values, mean2, "s-", color="steelblue",
             linewidth=2, markersize=5, label="V2 — context X")
    ax1.fill_between(rho_values, mean2 - std2, mean2 + std2,
                     alpha=0.2, color="steelblue")

    ax1.set_xlabel("Assumed ρ (Gaussian copula model)")
    ax1.set_ylabel("Estimated ATE")
    ax1.set_title("ρ-curve: mean ATE ± 1 std")
    ax1.legend(fontsize=9)

    # ── Right: RMSE ───────────────────────────────────────────────────────────
    ax2.axvline(REF_RHO, color="green", linestyle="--", linewidth=1.5,
                label=f"ρ_ref ≈ {REF_RHO:.3f}", zorder=1)

    ax2.plot(rho_values, rmse1, "o-", color="darkorange",
             linewidth=2, markersize=5, label="V1 — no context")
    ax2.plot(rho_values, rmse2, "s-", color="steelblue",
             linewidth=2, markersize=5, label="V2 — context X")

    for vals, color, label in [(rmse1, "darkorange", "V1"), (rmse2, "steelblue", "V2")]:
        best = int(np.argmin(vals))
        ax2.scatter(rho_values[best], vals[best], color=color, zorder=5, s=80,
                    label=f"{label} min @ ρ={rho_values[best]:.2f}")

    ax2.set_xlabel("Assumed ρ (Gaussian copula model)")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE vs Assumed ρ")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Setting — regime-switching copula\n"
        f"  X<0  → Clayton(θ={THETA_CLAYTON}),  τ={THETA_CLAYTON/(THETA_CLAYTON+2):.2f}\n"
        f"  X≥0  → Gumbel(θ={THETA_GUMBEL}),   τ={1-1/THETA_GUMBEL:.2f}\n"
        f"  ρ_ref = {REF_RHO:.4f}\n"
        f"  {len(RHO_VALUES)} ρ values × {N_SEEDS} seeds × 2 variants "
        f"= {len(RHO_VALUES)*N_SEEDS*2} runs\n"
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

    print(f"\n=== Summary  (true ATE = {TRUE_ATE}) ===")
    print(f"{'ρ':>8}  {'V1 mean':>9}  {'V1 RMSE':>8}  {'V2 mean':>9}  {'V2 RMSE':>8}")
    for i, rho in enumerate(RHO_VALUES):
        v1_mean = np.mean(est_v1[i])
        v1_rmse = np.sqrt(np.mean((est_v1[i] - TRUE_ATE) ** 2))
        v2_mean = np.mean(est_v2[i])
        v2_rmse = np.sqrt(np.mean((est_v2[i] - TRUE_ATE) ** 2))
        marker  = " ←" if abs(rho - REF_RHO) == min(abs(RHO_VALUES - REF_RHO)) else ""
        print(f"{rho:>8.3f}  {v1_mean:>9.4f}  {v1_rmse:>8.4f}  {v2_mean:>9.4f}  {v2_rmse:>8.4f}{marker}")

    np.savez(
        RESULTS_DIR / "results.npz",
        rho_values    = RHO_VALUES,
        est_v1        = est_v1,
        est_v2        = est_v2,
        true_ate      = TRUE_ATE,
        theta_clayton = THETA_CLAYTON,
        theta_gumbel  = THETA_GUMBEL,
        tau_mean      = TAU_MEAN,
        ref_rho       = REF_RHO,
    )
    print(f"Results saved → {RESULTS_DIR / 'results.npz'}")

    plot_results(RHO_VALUES, est_v1, est_v2, RESULTS_DIR / "comparison.png")


if __name__ == "__main__":
    main()
