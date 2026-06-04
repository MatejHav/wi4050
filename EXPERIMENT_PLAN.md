Based on th# Experiment Plan: Testing Gaussian Copula Assumption in ρ-GNF

## Motivation

The ρ-GNF model estimates the Average Treatment Effect (ATE) under unobserved confounding
by assuming that the latent noise variables `(z_A, z_Y)` follow a multivariate Gaussian with
covariance `Z_Sigma(ρ)`. The correlation ρ parameterises the non-causal association between
treatment and outcome due to the hidden confounder.

This plan investigates two questions:

1. **How badly does the model break** when the true confounding departs from a Gaussian copula?
2. **Does extending the model** to accept observed confounders and a conditional copula recover
   correct ATE estimates?

---

## Part 0 — Implementation Extensions (Prerequisites)

These changes to the codebase must be completed before the experiments in Parts 1 and 2.

### 0a — Wire context into DAGConditioner

**File:** `models/Conditionners/DAGConditioner.py`

**Problem:** The `context` parameter is accepted in the method signature but silently dropped in
both forward branches. The `cond_in` parameter in `DAGMLP` already widens the input layer to
accommodate extra inputs, but nothing is ever passed in.

**Change:** After computing the masked embedding `e` (the A-matrix-masked input), concatenate
context before passing to the embedding net:

```python
if context is not None:
    context_expanded = context.unsqueeze(1).expand(-1, self.in_size, -1)
    e = torch.cat((e, context_expanded), -1)
```

Apply this in both the hot-encoding and non-hot-encoding branches. Also update
`buildFCNormalizingFlow_UC` in `NormalizingFlowFactories.py` to accept and forward a
`cond_in` argument, and thread `context` through all call sites in the training loop.

**Effect:** Observed confounders passed as `context` will condition each variable's normalizing
transformation on those covariates, directly adjusting for observed confounding within the
flow.

### 0b — Conditional copula: ρ as a function of observed covariates

**File:** `models/NormalizingFlowFactories.py` (`NormalLogDensity_UC`)

**Problem:** `Z_Sigma` is a fixed global matrix. The scalar ρ is the same for every observation,
so the model cannot express that confounding strength varies across subgroups.

**Change:** Replace the fixed ρ with a small neural network `ρ_net(context) → ρ(x) ∈ (−1, +1)`:

```python
class ConditionalNormalLogDensity_UC(nn.Module):
    def __init__(self, base_Z_Sigma, cond_dim):
        # rho_net: context -> scalar rho per sample
        self.rho_net = nn.Sequential(
            nn.Linear(cond_dim, 16), nn.Tanh(),
            nn.Linear(16, 1), nn.Tanh()   # output in (-1, +1)
        )

    def forward(self, z, context):
        rho = self.rho_net(context)  # [B, 1]
        # construct per-sample Z_Sigma from rho and evaluate log prob
        ...
```

Per-sample `Z_Sigma_i = [[1, ρ(x_i)], [ρ(x_i), 1]]` is used to evaluate
`log p(z_i | x_i)` individually. The loss becomes:
`E[log p_Z(z(x) | x) + log|det J(x)|] + constraintsLoss()`

The original fixed-ρ model is a special case (constant network), so all existing experiments
continue to run unchanged.

---

## Part 1 — Baseline Correctness Evaluation

### Protocol

For every experimental setting:

- Generate `K = 100` independent datasets of `n = 5,000` samples each
- For each dataset, train the ρ-GNF with the true ρ (or closest approximation)
- Record `ATE_estimated` from the Monte Carlo flow-inversion procedure
- Compute the error `ATE_estimated − ATE_true` (true ATE is known analytically from the DGP)
- Aggregate the K errors into a distribution

Fixed parameters across all settings unless otherwise noted:
- `α = 0.2` (true ATE)
- `n_mce_samples = 2000` (Monte Carlo samples for ATE estimation)
- Same model architecture as the existing `ToySimulatedContinuous.py`

---

## Part 2 — Experimental Settings

### Setting 0 — Baseline (assumption satisfied)

**DGP:**
```
(e_A, e_Y) ~ MVN(0, [[1, β], [β, δ]])
A = e_A
Y = α·A + e_Y
```
With `β = −0.6`, `δ = 0.72`, matching the existing Hoover model (`ToySimulatedContinuous.py`).
The model is given the correct ρ.

**Purpose:** Control condition. Establishes the noise floor and validates the evaluation
protocol. Should show near-zero bias.

**True ATE:** α = 0.2

---

### Setting 1 — Heavy tails (t-copula)

**DGP:**
```
(u_A, u_Y) ~ Bivariate Student-t(ν, ρ)     # same rank correlation as baseline
A = Φ^{-1}(F_t(u_A))                       # map to Gaussian marginals
Y = α·A + Φ^{-1}(F_t(u_Y))
```
Run with `ν ∈ {2, 3, 5, 10}`. The model is given the matched ρ but still assumes a Gaussian
copula.

**Purpose:** Tests sensitivity to tail dependence. Gaussian and t-copulas share the same
linear correlation but the t-copula has stronger co-occurrence of extreme values.

**True ATE:** α = 0.2

---

### Setting 2 — Asymmetric tail dependence (Clayton copula)

**DGP:**
```
(u_A, u_Y) ~ Clayton copula(θ)     # lower tail dependence, near-independence in upper tail
A = u_A,   Y = α·A + u_Y
```
θ chosen so that Kendall's τ matches the baseline. Model is given the equivalent Gaussian ρ
(matched via Kendall's τ → ρ conversion).

**Purpose:** Tests copula shape misspecification. The Clayton copula is structurally different
from Gaussian — the model's symmetric Gaussian copula cannot represent asymmetric tail
dependence regardless of ρ.

**True ATE:** α = 0.2

---

### Setting 3 — Non-Gaussian marginals, Gaussian copula (positive control)

**DGP:**
```
(u_A, u_Y) ~ MVN(0, Z_Sigma(ρ))    # Gaussian copula, correct assumption
A = exp(u_A)                         # log-normal marginal
Y = α·A + u_Y^3                      # heavy-tailed marginal
```
Model is given the correct ρ.

**Purpose:** Positive control. Since the flow learns monotone marginal transforms, it should
handle non-Gaussian marginals while the copula remains Gaussian. This setting should behave
similarly to Setting 0.

**True ATE:** α (computed via simulation from the DGP)

---

### Setting 4 — Discrete hidden confounder

**DGP:**
```
U ~ Bernoulli(0.5)
A = γ_A·U + ε_A,    ε_A ~ N(0, 1)
Y = α·A + γ_Y·U + ε_Y,    ε_Y ~ N(0, 1)
```
With `γ_A = γ_Y = 1`. No copula of any kind can represent binary confounding exactly.
Model is given ρ matched to the rank correlation of `(A, Y)` induced by U.

**Purpose:** Most realistic violation. Discrete confounders are common in practice (e.g.
binary policy, subgroup membership). Tests whether the Gaussian copula framework fails
categorically or degrades gracefully.

**True ATE:** α = 0.2

---

### Setting 5 — Non-linear confounding path

**DGP:**
```
U ~ N(0, 1)
A = U + ε_A,    ε_A ~ N(0, 0.5)
Y = α·A + U² + ε_Y,    ε_Y ~ N(0, 0.5)
```
U enters Y quadratically. No choice of ρ can capture this because the copula only
models the rank dependence, not the functional form.

**Purpose:** Tests whether non-linearity in the confounding mechanism causes bias even
when the marginal dependence is correctly matched by ρ.

**True ATE:** α = 0.2 (since `E[U²]` is constant and cancels in `E[Y|do(A=1)] − E[Y|do(A=0)]`)

---

### Setting 6 — Heterogeneous confounding (conditional copula stress test)

**DGP:**
```
U ~ Bernoulli(0.5)    [observed]
(e_A, e_Y) | U=0 ~ MVN(0, Z_Sigma(ρ=0.1))
(e_A, e_Y) | U=1 ~ MVN(0, Z_Sigma(ρ=0.7))
A = e_A,   Y = α·A + e_Y
```
Confounding strength differs strongly across the two groups. U is observed and passed as
context.

**Purpose:** Specifically designed for the conditional copula extension (Task 0b). A global
fixed ρ will be wrong for both subgroups simultaneously. The conditional copula `ρ(U)` should
recover the correct subgroup ATEs.

**True ATE:** α = 0.2 (same in both groups by construction)

---

## Part 3 — Model Variants

Every setting (except Setting 0 which has no observed U) is run under three model variants:

| Variant | Description |
|---|---|
| **V1 — original** | Fixed ρ, no context, Gaussian copula (existing model) |
| **V2 — +context** | Observed U passed as context into DAGConditioner (Task 0a) |
| **V3 — +conditional copula** | ρ(U) network learns confounding strength per covariate (Task 0b) |

This isolates two effects independently:
- **V2 vs V1**: Does directly conditioning the flow on U fix the bias?
- **V3 vs V1**: Does a covariate-dependent ρ help when confounding is heterogeneous?

---

## Part 4 — Plots

### Plot 1 — Baseline calibration
Histogram of `(ATE_estimated − ATE_true)` across 100 runs for Setting 0 only.
Should be centred at 0. Validates the protocol and establishes the noise floor.

### Plot 2 — Error distributions across settings (V1 only)
Violin plots of `(ATE_estimated − ATE_true)` for Settings 0–5 under V1 (original model).
Arranged left to right from "assumption satisfied" to "most violated."
Dashed horizontal line at 0. Main result showing where the Gaussian copula assumption breaks.

### Plot 3 — Three-way comparison (V1 vs V2 vs V3)
For each setting, three violins side by side (V1, V2, V3).
Shows whether the implementation extensions recover correct ATE estimation.
Primary result figure for the extension evaluation.

### Plot 4 — ρ-curve per setting
For one representative dataset per setting, plot the full ρ-curve
(`ATE_estimated` vs `ρ ∈ [−1, +1]`).
- Horizontal dashed line: true ATE
- Vertical dashed line: assumed ρ
- Shows whether the true ATE is even on the ρ-curve, and how sensitive the estimate is to ρ
  misspecification under each DGP.

### Plot 5 — ρ-curve with and without context (Settings 4 and 5)
Overlay ρ-curves from V1 and V2 on the same axes for Settings 4 and 5.
If conditioning on U collapses the curve to a single correct ATE regardless of ρ,
identification is recovered by observation alone.

### Plot 6 — Bias as a function of tail weight (Setting 1)
For the t-copula setting, vary `ν ∈ {2, 3, 5, 10, ∞}` and plot mean bias ± std as a
function of ν. Shows how quickly the estimate degrades as tails get heavier.

### Plot 7 — Learned ρ(U) vs true ρ(U) (Setting 6, V3 only)
Scatter of `ρ_net(u_i)` against the true subgroup ρ (0.1 or 0.7).
Shows whether the conditional copula network recovers the correct confounding structure per
subgroup.

### Plot 8 — Estimated vs true ATE scatter
One point per dataset per setting, `ATE_true` on x-axis, `ATE_estimated` on y-axis,
coloured by setting. Diagonal = perfect calibration.
Deviation from diagonal shows systematic bias; spread shows variance.

---

## Summary Table (to be filled after experiments)

| Setting | Copula | Marginals | Confounder | V1 Bias | V2 Bias | V3 Bias |
|---|---|---|---|---|---|---|
| 0 | Gaussian | Gaussian | None | ~0 | — | — |
| 1 (ν=3) | t | Gaussian | None | ? | — | — |
| 2 | Clayton | Gaussian | None | ? | — | — |
| 3 | Gaussian | Non-Gaussian | None | ~0 | — | — |
| 4 | None | Mixed | Binary U | ? | ? | ? |
| 5 | None | Gaussian | Non-linear U | ? | ? | ? |
| 6 | Heterogeneous | Gaussian | Binary U (obs.) | ? | ? | ? |
