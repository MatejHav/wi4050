"""
Setting: Piecewise Gaussian Copula with confounder X, binary treatment A, outcome Y.

DGP (Structural Causal Model)
------------------------------
    X   ~ Uniform(-1, 1)                        observed confounder
    Z_A ~ N(0, 1)                               latent treatment noise  [unobserved]
    Z_Y | Z_A, X ~ N(rho(X)*Z_A, sqrt(1-rho(X)^2))   latent outcome noise [unobserved]
         where rho(X) = RHO_NEG if X < 0
                        RHO_POS if X >= 0
    A   = Bernoulli(sigmoid(X + Z_A))           binary treatment
    Y   = Z_Y + TRUE_ATE * A                    continuous outcome

Confounding: X shifts both the propensity score of A (via sigmoid(X + Z_A)) and
the latent copula correlation between Z_A and Z_Y.  Because Z_Y feeds directly
into Y, and Z_Y is correlated with Z_A (which drives A), naive regression
E[Y|A=1] - E[Y|A=0] != TRUE_ATE.

True ATE (analytical)
---------------------
Under do(A=a) the link X->A is cut.  X remains Uniform(-1,1), Z_A and Z_Y
are drawn from their joint (BivariateNormal with rho(X)), and A is set to a:
    E[Y | do(A=a)] = E[Z_Y] + TRUE_ATE * a = 0 + TRUE_ATE * a
    ATE = E[Y|do(A=1)] - E[Y|do(A=0)] = TRUE_ATE

Experiment
----------
The rho-GNF model is trained on (A, Y) pairs with X supplied as external
rho context.  Two variants are compared:

  - split rho (oracle): two fixed-rho models on X<0 / X>=0 subsets -> recovers TRUE_ATE
  - fixed rho = avg   : constant rho=0.15                          -> biased ATE
  - fixed rho = 0     : no correlation                             -> biased ATE

Output
------
    results/setting_piecewise_copula/results.npz
    results/setting_piecewise_copula/ate_figure.png
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
    build_rho_fn_gnf,
    train_model,
    estimate_ate,
)

# ── DGP parameters ────────────────────────────────────────────────────────────
RHO_NEG  = -0.5   # copula rho between Z_A and Z_Y when X < 0
RHO_POS  =  0.8   # copula rho between Z_A and Z_Y when X >= 0
RHO_AVG  = (RHO_NEG + RHO_POS) / 2.0   # = 0.15

TRUE_ATE = 1 # structural coefficient of A in Y = Z_Y + TRUE_ATE * A

# Reference intervention contrast (binary treatment)
A0 = 0.0
A1 = 1.0

TRUE_EY0 = 0.0        # E[Y | do(A=0)] = E[Z_Y] + TRUE_ATE*0 = 0
TRUE_EY1 = TRUE_ATE   # E[Y | do(A=1)] = E[Z_Y] + TRUE_ATE*1 = TRUE_ATE

# cat_dims for the normalizer: column 0 (A) is binary (2 categories)
CAT_DIMS = {0: 2}

# ── Experiment parameters ─────────────────────────────────────────────────────
N_SEEDS   = 20
N_SAMPLES = 10_000

CONFIG = dict(
    emb_net         = [20, 15, 10],
    int_net         = [15, 10, 5],
    nb_steps        = 50,
    solver          = "CC",
    nb_flow         = 1,
    l1              = 0.5,
    b_size          = 128,
    nb_epoch        = 10,
    learning_rate   = 1e-2,
    nb_estop        = 10,
    nb_epoch_update = 50,
    n_mce_samples   = 2_000,
    mce_b_size      = 2_000,
)

RESULTS_DIR = Path(__file__).parent / "results" / "setting_piecewise_copula"


# ── rho function ──────────────────────────────────────────────────────────────

def rho_fn(x: torch.Tensor) -> torch.Tensor:
    """True piecewise rho: RHO_NEG for X<0, RHO_POS for X>=0. Input/output [B,1]."""
    return torch.where(x < 0,
                       torch.full_like(x, RHO_NEG),
                       torch.full_like(x, RHO_POS))


# ── Data generation ───────────────────────────────────────────────────────────

def generate_data(n_samples, seed):
    """Returns FloatTensor [n_samples, 3] with columns (X, A, Y)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x   = torch.empty(n_samples, 1).uniform_(-1, 1)
    z_a = torch.randn(n_samples, 1)

    rho_x = torch.where(x < 0, torch.full_like(x, RHO_NEG), torch.full_like(x, RHO_POS))
    z_y   = rho_x * z_a + torch.sqrt(1.0 - rho_x ** 2) * torch.randn(n_samples, 1)

    a = torch.bernoulli(torch.sigmoid(x + z_a))
    y = z_y + TRUE_ATE * a

    return torch.cat([x, a, y], dim=1)


# ── Single run helpers ────────────────────────────────────────────────────────

def _prepare(seed):
    """Split data into model input (A, Y) and rho context (X)."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_all = generate_data(N_SAMPLES, seed)   # [N, 3]: (X, A, Y)
    data_x   = data_all[:, 0:1]                 # [N, 1]: X  (rho context)
    data_ay  = data_all[:, 1:]                  # [N, 2]: (A, Y) for model

    data_ay_train, data_ay_val = split_data(data_ay)
    data_x_train,  data_x_val  = split_data(data_x)

    data_mu, data_sigma = compute_normalisation(data_ay_train, data_ay_val)

    return data_ay_train, data_ay_val, data_x_train, data_x_val, \
           data_mu, data_sigma, device


def run_rho_fn(seed):
    """Train with the true piecewise rho_fn(X) and return ATE estimate."""
    data_ay_train, data_ay_val, data_x_train, data_x_val, \
        data_mu, data_sigma, device = _prepare(seed)

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
        cat_dims        = CAT_DIMS,
    )
    model, _ = train_model(
        model, data_ay_train, data_ay_val,
        nb_epoch          = CONFIG["nb_epoch"],
        b_size            = CONFIG["b_size"],
        nb_steps          = CONFIG["nb_steps"],
        learning_rate     = CONFIG["learning_rate"],
        nb_estop          = CONFIG["nb_estop"],
        device            = device,
        rho_context_train = data_x_train,
        rho_context_val   = data_x_val,
    )

    Z_Sigma_id = get_cov_matrix(0.0)
    return estimate_ate(
        model,
        Z_Sigma        = Z_Sigma_id,
        n_mce_samples  = CONFIG["n_mce_samples"],
        mce_b_size     = CONFIG["mce_b_size"],
        device         = device,
        treatment_dim  = 0,
        treatment_vals = (A0, A1),
    )


def run_split_rho(seed):
    """Train two fixed-rho models on X<0 / X>=0 subsets and return combined ATE."""
    data_ay_train, data_ay_val, data_x_train, data_x_val, \
        _, _, device = _prepare(seed)

    mask_neg_train = data_x_train[:, 0] < 0
    mask_pos_train = ~mask_neg_train
    mask_neg_val   = data_x_val[:, 0] < 0
    mask_pos_val   = ~mask_neg_val

    def _train_and_estimate(ay_tr, ay_va, rho):
        mu, sigma = compute_normalisation(ay_tr, ay_va)
        Z_Sigma = get_cov_matrix(rho)
        model = build_rho_gnf(
            Z_Sigma         = Z_Sigma,
            emb_net         = CONFIG["emb_net"],
            int_net         = CONFIG["int_net"],
            nb_steps        = CONFIG["nb_steps"],
            solver          = CONFIG["solver"],
            l1              = CONFIG["l1"],
            nb_flow         = CONFIG["nb_flow"],
            data_mu         = mu,
            data_sigma      = sigma,
            nb_epoch_update = CONFIG["nb_epoch_update"],
            device          = device,
            cat_dims        = CAT_DIMS,
        )
        model, _ = train_model(
            model, ay_tr, ay_va,
            nb_epoch      = CONFIG["nb_epoch"],
            b_size        = CONFIG["b_size"],
            nb_steps      = CONFIG["nb_steps"],
            learning_rate = CONFIG["learning_rate"],
            nb_estop      = CONFIG["nb_estop"],
            device        = device,
        )
        # The latent z_A encoding of A=0/1 is calibrated to this stratum's P(A=1),
        # so we must recover the actual z_A values rather than using (0.0, 1.0).
        # z_A is a root node (no DAG parents), so any Y value gives the same z_A.
        model.eval()
        mean_y = float(mu[1].item())
        with torch.no_grad():
            z_a0, _ = model(torch.tensor([[A0, mean_y]], dtype=torch.float32, device=device))
            z_a1, _ = model(torch.tensor([[A1, mean_y]], dtype=torch.float32, device=device))
            tv0 = float(z_a0[0, 0].item())
            tv1 = float(z_a1[0, 0].item())

        ate = estimate_ate(
            model,
            Z_Sigma        = Z_Sigma,
            n_mce_samples  = CONFIG["n_mce_samples"],
            mce_b_size     = CONFIG["mce_b_size"],
            device         = device,
            treatment_dim  = 0,
            treatment_vals = (tv0, tv1),
        )
        return ate, len(ay_tr) + len(ay_va)

    ate_neg, n_neg = _train_and_estimate(
        data_ay_train[mask_neg_train], data_ay_val[mask_neg_val], RHO_NEG
    )
    ate_pos, n_pos = _train_and_estimate(
        data_ay_train[mask_pos_train], data_ay_val[mask_pos_val], RHO_POS
    )

    return (n_neg * ate_neg + n_pos * ate_pos) / (n_neg + n_pos)


def run_fixed_rho(assumed_rho, seed):
    """Train with a constant assumed_rho and return ATE estimate."""
    data_ay_train, data_ay_val, _, _, \
        data_mu, data_sigma, device = _prepare(seed)

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
        cat_dims        = CAT_DIMS,
    )
    model, _ = train_model(
        model, data_ay_train, data_ay_val,
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
        treatment_vals = (A0, A1),
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(est_rho0, est_rho_avg, est_split, save_path):
    n_seeds = len(est_rho0)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"Piecewise Copula — X confounder, A binary treatment, Y outcome\n"
        f"rho(X<0)={RHO_NEG}  rho(X>=0)={RHO_POS}  "
        f"TRUE_ATE={TRUE_ATE:.1f}  (n={N_SAMPLES}, {n_seeds} seeds)",
        fontsize=10,
    )

    variants = [
        (est_rho0,    "fixed rho=0",                    "purple",      "^"),
        (est_rho_avg, f"fixed rho={RHO_AVG:.2f} (avg)", "darkorange",  "s"),
        (est_split,   "split rho (oracle)",              "forestgreen", "D"),
    ]

    jitter = np.linspace(-0.12, 0.12, n_seeds)
    for k, (ests, label, color, marker) in enumerate(variants):
        xpos = k + 1
        ax.scatter(xpos + jitter, ests, color=color, marker=marker,
                   alpha=0.6, zorder=3, s=40,
                   label=f"{label}  (mean={np.mean(ests):.3f})")
        ax.errorbar(xpos, np.mean(ests), yerr=np.std(ests),
                    fmt="none", color=color, capsize=6, linewidth=2, zorder=4)

    ax.axhline(TRUE_ATE, color="red", linestyle="--", linewidth=1.5,
               label=f"True ATE = {TRUE_ATE:.1f}", zorder=2)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(
        ["fixed rho=0\n(misspec.)",
         f"fixed rho={RHO_AVG:.2f}\n(misspec.)",
         "split rho\n(oracle)"],
        fontsize=9,
    )
    ax.set_ylabel("Estimated ATE  (do(A=1) - do(A=0))")
    ax.set_title("ATE estimates — dots = individual seeds, bar = mean +/- 1 std")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Setting - piecewise copula  (X confounder, A binary, Y outcome)\n"
        f"  X   ~ Uniform(-1,1)\n"
        f"  Z_A ~ N(0,1)  [unobserved]\n"
        f"  Z_Y | Z_A, X ~ N(rho(X)*Z_A, sqrt(1-rho(X)^2))  [unobserved]\n"
        f"       rho(X) = {RHO_NEG} (X<0)  |  {RHO_POS} (X>=0)\n"
        f"  A   = Bernoulli(sigmoid(X + Z_A))\n"
        f"  Y   = Z_Y + {TRUE_ATE} * A\n"
        f"\n"
        f"  True ATE = {TRUE_ATE}  ({N_SEEDS} seeds, n={N_SAMPLES})\n"
    )

    est_rho0    = np.zeros(N_SEEDS)
    est_rho_avg = np.zeros(N_SEEDS)
    est_split   = np.zeros(N_SEEDS)

    for s in range(N_SEEDS):
        print(f"[seed={s}]  fixed rho=0", end="  ", flush=True)
        est_rho0[s] = run_fixed_rho(assumed_rho=0.0, seed=s)
        print(f"ATE={est_rho0[s]:+.3f}  |  fixed rho={RHO_AVG:.2f}", end="  ", flush=True)
        est_rho_avg[s] = run_fixed_rho(assumed_rho=RHO_AVG, seed=s)
        print(f"ATE={est_rho_avg[s]:+.3f}  |  split rho", end="  ", flush=True)
        est_split[s] = run_split_rho(seed=s)
        print(f"ATE={est_split[s]:+.3f}", flush=True)

    print(f"\n=== Summary  (true ATE = {TRUE_ATE}) ===")
    print(f"{'Variant':<30}  {'Mean':>8}  {'Bias':>8}  {'RMSE':>8}")
    for label, ests in [
        ("fixed rho=0",                    est_rho0),
        (f"fixed rho={RHO_AVG:.2f} (avg)", est_rho_avg),
        ("split rho (oracle)",             est_split),
    ]:
        mean = float(np.mean(ests))
        bias = float(np.mean(ests - TRUE_ATE))
        rmse = float(np.sqrt(np.mean((ests - TRUE_ATE) ** 2)))
        print(f"{label:<30}  {mean:>8.3f}  {bias:>8.3f}  {rmse:>8.3f}")

    np.savez(
        RESULTS_DIR / "results.npz",
        est_rho0    = est_rho0,
        est_rho_avg = est_rho_avg,
        est_split   = est_split,
        true_ate    = TRUE_ATE,
        true_ey0    = TRUE_EY0,
        true_ey1    = TRUE_EY1,
        rho_neg     = RHO_NEG,
        rho_pos     = RHO_POS,
        rho_avg     = RHO_AVG,
        a0          = A0,
        a1          = A1,
    )
    print(f"Results saved -> {RESULTS_DIR / 'results.npz'}")

    plot_results(est_rho0, est_rho_avg, est_split, RESULTS_DIR / "ate_figure.png")


if __name__ == "__main__":
    main()
