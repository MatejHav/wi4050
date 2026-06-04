"""
Setting 0 & 1 — observed Y vs adjusted (interventional) Y distributions.

Two-row grid (Setting 0 = Gaussian copula, Setting 1 = Clayton copula).
Seven columns, one per rho value in RHO_VALUES.

Each cell shows:
  • Orange histogram  — observed marginal p(Y)  (confounded, from data)
  • Blue density      — p(Y | do(A=0)) = N(0, 1)  \
  • Green density     — p(Y | do(A=1)) = N(α, 1)  / true interventional targets

The gap between blue and green means is the true ATE = α = 1.

These interventional distributions are what the rho-GNF approximates via:
    sample Z ~ N(0, Z_Σ),  pin Z_A = a,  invert flow → Y | do(A = a).
With a correctly specified model they match N(α·a, 1); copula misspecification
(Setting 1) distorts this recovery without changing the theoretical target.

For Setting 1, each rho is mapped to a Clayton theta via shared Kendall's τ:
    τ = (2/π) arcsin(ρ)    [Gaussian identity]
    θ = 2τ / (1 – τ)       [Clayton identity]

Output
------
    results/setting01_zy_distribution/zy_distribution.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import norm as scipy_norm

# ── Parameters ────────────────────────────────────────────────────────────────
RHO_VALUES = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
ALPHA      = 1.0        # true ATE
N_SAMPLES  = 20_000
SEED       = 42

RESULTS_DIR = Path(__file__).parent / "results" / "setting01_zy_distribution"


# ── Copula helpers ────────────────────────────────────────────────────────────

def rho_to_kendall_tau(rho: float) -> float:
    return float((2.0 / np.pi) * np.arcsin(np.clip(rho, -1 + 1e-9, 1 - 1e-9)))


def rho_to_clayton_theta(rho: float) -> float:
    tau = rho_to_kendall_tau(rho)
    if abs(tau) < 1e-9:
        return 0.0
    return float(np.clip(2.0 * tau / (1.0 - tau), -1.0 + 1e-6, 50.0))


# ── Data generation — Setting 0 (Gaussian copula) ────────────────────────────

def generate_gaussian(n: int, alpha: float, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rho = float(np.clip(rho, -1 + 1e-9, 1 - 1e-9))
    noise = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], n)
    e_A, e_Y = noise[:, 0], noise[:, 1]
    return (alpha * e_A + e_Y).astype(np.float32)   # observed Y


# ── Data generation — Setting 1 (Clayton copula, N(0,1) margins) ─────────────

def sample_clayton_copula(n: int, theta: float, rng) -> tuple:
    """Conditional-CDF inversion.  Handles edge cases at theta → {0, -1, +∞}."""
    if theta >= 50.0:
        u1 = rng.uniform(0.0, 1.0, n)
        return u1, u1.copy()
    if theta <= -1.0 + 1e-5:
        u1 = rng.uniform(0.0, 1.0, n)
        return u1, 1.0 - u1
    if abs(theta) < 1e-6:
        return rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)

    u1 = rng.uniform(0.0, 1.0, n)
    v  = rng.uniform(0.0, 1.0, n)
    inner = u1 ** (-theta) * (v ** (-theta / (theta + 1.0)) - 1.0) + 1.0
    u2 = np.clip(np.maximum(inner, 1e-300) ** (-1.0 / theta), 1e-7, 1 - 1e-7)
    return u1, u2


def generate_clayton(n: int, alpha: float, theta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u1, u2 = sample_clayton_copula(n, theta, rng)
    e_A = scipy_norm.ppf(u1).astype(np.float32)
    e_Y = scipy_norm.ppf(u2).astype(np.float32)
    return alpha * e_A + e_Y                          # observed Y


# ── Plotting ──────────────────────────────────────────────────────────────────

def _format_theta(theta: float) -> str:
    if theta >= 50.0:    return "+inf"
    if theta <= -1 + 1e-4: return "-1"
    return f"{theta:+.3f}"


def plot_distributions(save_path: Path):
    n_rho  = len(RHO_VALUES)
    x_lo, x_hi = -5.0, 6.0
    x_ref  = np.linspace(x_lo, x_hi, 500)

    # True interventional densities — same for both settings
    pdf_do0 = scipy_norm.pdf(x_ref, loc=0.0,   scale=1.0)   # do(A=0)
    pdf_do1 = scipy_norm.pdf(x_ref, loc=ALPHA,  scale=1.0)   # do(A=1)

    fig = plt.figure(figsize=(3.2 * n_rho, 6.5))
    gs  = gridspec.GridSpec(2, n_rho, figure=fig, hspace=0.55, wspace=0.25)

    bins = np.linspace(x_lo, x_hi, 70)

    row_labels = ["Setting 0\n(Gaussian copula)", "Setting 1\n(Clayton copula)"]

    for col, rho in enumerate(RHO_VALUES):
        theta = rho_to_clayton_theta(rho)
        tau   = rho_to_kendall_tau(rho)

        Y_g = generate_gaussian(N_SAMPLES, ALPHA, rho,   seed=SEED + col)
        Y_c = generate_clayton( N_SAMPLES, ALPHA, theta, seed=SEED + col)

        for row, (Y_obs, extra_title) in enumerate([
            (Y_g, f"ρ = {rho:+.2f}"),
            (Y_c, f"ρ = {rho:+.2f}  |  θ = {_format_theta(theta)},  τ = {tau:+.3f}"),
        ]):
            ax = fig.add_subplot(gs[row, col])

            # Observed Y — non-parametric
            ax.hist(Y_obs, bins=bins, density=True, alpha=0.50,
                    color="tab:orange", label="Y observed")

            # Adjusted distributions — interventional targets
            ax.plot(x_ref, pdf_do0, color="tab:blue",  linewidth=1.8,
                    label="do(A=0)  N(0,1)")
            ax.plot(x_ref, pdf_do1, color="tab:green", linewidth=1.8,
                    label=f"do(A=1)  N({ALPHA:.0f},1)")

            # ATE arrow between the two interventional means
            y_arrow = 0.45 * max(pdf_do0.max(), pdf_do1.max())
            ax.annotate("", xy=(ALPHA, y_arrow), xytext=(0.0, y_arrow),
                        arrowprops=dict(arrowstyle="<->", color="black",
                                        lw=1.2))
            ax.text(ALPHA / 2, y_arrow + 0.015, f"ATE={ALPHA:.0f}",
                    ha="center", va="bottom", fontsize=6.5)

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(bottom=0)
            ax.set_title(extra_title, fontsize=7.5, pad=3)
            ax.tick_params(labelsize=7)

            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8.5)
                if row == 0:
                    ax.legend(fontsize=6, loc="upper left",
                              framealpha=0.7, handlelength=1.2)

    fig.suptitle(
        f"Observed Y (orange) vs true interventional targets — "
        f"α = {ALPHA:.0f},  n = {N_SAMPLES:,}\n"
        f"Blue/green = p(Y|do(A=a))=N(αa, 1): what ATE estimation adjusts for.  "
        f"Confounding (ρ) shifts the orange histogram away from the targets.",
        fontsize=9.5, y=1.02,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_distributions(RESULTS_DIR / "zy_distribution.png")


if __name__ == "__main__":
    main()
