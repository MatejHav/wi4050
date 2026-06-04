# ρ-GNF: Sensitivity Analysis to Unobserved Confounding via Normalizing Flows

---

## Presentation Structure

**Slides (14 content frames):**

| Section | Slides | Topic |
|---|---|---|
| Causal Inference | 1–4 | Smoking/cancer example → ignorability → ATE → sensitivity motivation |
| ρ-GNF Model | 5–7 | Gaussian copula as ρ · generative model · ATE estimator |
| Experiments | 8–11 | Setting 0 (Gaussian, sanity check) · Setting 1 (Clayton, misspecification) |
| Limitations | 12–13 | Gaussian copula assumption · ρ blind to covariates |
| Extension | 14 | Context conditioning V1 vs V2 · Setting 2 results pending |

**Backup slides** (after Q&A, not counted): DAGConditioner · UMNN · Training objective. Shown only if questions arise; the audience needs only that $f_\theta$ is "learned end-to-end via maximum likelihood."

---

## 1. Causal Inference and the Problem of Confounding

### 1.1 The Goal: Average Treatment Effect

Causal inference asks a fundamentally different question from prediction. Given observational data on treatment $A$ and outcome $Y$, we want to know what would happen to $Y$ if we **intervened** to set $A = a$, rather than what $Y$ tends to be when $A = a$ is passively observed.

The **Average Treatment Effect (ATE)** formalises this:

$$\text{ATE} = \mathbb{E}[Y \mid \text{do}(A = 1)] - \mathbb{E}[Y \mid \text{do}(A = 0)]$$

Using Pearl's do-calculus, $\text{do}(A = a)$ means surgically setting $A$ to $a$ by removing all arrows into $A$ in the causal graph — it is not the same as conditioning on $A = a$.

**Running example (used in slides 1–3):** Does smoking cause lung cancer? Smokers have higher cancer rates, but this could be explained by lifestyle — heavy smokers may also drink more, exercise less, and eat differently. We need a causal framework to separate the direct effect from these confounders.

### 1.2 Confounding

In observational data the naive regression estimate $\mathbb{E}[Y \mid A=1] - \mathbb{E}[Y \mid A=0]$ is biased whenever there exists an **unobserved common cause** $U$ of both $A$ and $Y$:

```
      U
    ↙   ↘
  A   →   Y
```

$U$ opens a **backdoor path** $A \leftarrow U \rightarrow Y$. The observed correlation between $A$ and $Y$ mixes the causal effect $\alpha$ with the confounding signal through $U$.

In the smoking example, $U$ is "lifestyle" — unobserved habits that simultaneously push people toward smoking and toward worse cancer outcomes.

### 1.3 Conditional Ignorability

The standard solution to confounding is **conditional ignorability** (also called *no unmeasured confounders*):

$$Y(a) \perp\!\!\!\perp A \;\mid\; \mathbf{X} \qquad \text{for all } a$$

where $\mathbf{X}$ are observed pre-treatment covariates and $Y(a)$ is the potential outcome under $\text{do}(A = a)$.

If this holds, the ATE can be recovered via **adjustment**:

$$\text{ATE} = \mathbb{E}_{\mathbf{X}}\!\left[\mathbb{E}[Y \mid A=1, \mathbf{X}] - \mathbb{E}[Y \mid A=0, \mathbf{X}]\right]$$

In the smoking example: if we measure diet, SES, and exercise, and those capture all confounding, then within each stratum of $(X_1, X_2, X_3)$ the smoking–cancer association becomes causal.

**The fundamental limitation:** conditional ignorability cannot be verified from data. It asserts that all confounders are measured. In practice, important confounders are routinely unmeasured.

---

## 2. Sensitivity Analysis

### 2.1 Motivation

When ignorability is violated, the question shifts from *"what is the ATE?"* to *"how sensitive is the ATE estimate to the strength of unmeasured confounding?"*

Sensitivity analysis does not assume ignorability. Instead it asks:

> How strong would unmeasured confounding have to be to explain away (nullify, reverse) the observed effect?

If the answer is "very strong", the effect is said to be *robust* to confounding. If the answer is "mild", the conclusion is fragile.

### 2.2 The Copula-Based Approach

A more expressive alternative models the **joint distribution of the confounding noise** using a **copula**. The copula separates marginal distributions (which can be estimated from data) from the dependence structure (which cannot, due to $U$ being unobserved).

A **Gaussian copula** parametrises this residual dependence by a single number $\rho \in (-1, +1)$:

$$C_\rho(u_A, u_Y) = \Phi_\rho\!\left(\Phi^{-1}(u_A),\, \Phi^{-1}(u_Y)\right)$$

where $\Phi_\rho$ is the bivariate standard normal CDF with correlation $\rho$.

The sensitivity analysis then becomes: **sweep $\rho$ from $-1$ to $+1$ and observe how the ATE estimate changes.** The value $\rho = 0$ is the ignorability assumption; the value at which the ATE crosses zero is the *critical* confounding strength needed to explain the effect away.

---

## 3. The ρ-GNF Framework

### 3.1 Core Idea

**ρ-GNF** (rho-Graphical Normalizing Flow) implements this copula-based sensitivity analysis using a normalizing flow. The key contributions are:

1. The marginal distributions of $A$ and $Y$ are modelled **non-parametrically** by the flow — no linearity or Gaussian assumption.
2. The confounding dependence is modelled **separately** through the Gaussian copula with parameter $\rho$.
3. The ATE is estimated by a **Monte Carlo inversion** of the learned flow, which is differentiable and exact (no numerical grid).

### 3.2 From Copula to Model

By Sklar's theorem, any joint distribution with marginal CDFs $F_A$ and $F_Y$ can be written as:

$$p(A, Y) = f_A(A) \cdot f_Y(Y) \cdot c_\rho\!\left(F_A(A),\, F_Y(Y)\right)$$

where $c_\rho$ is the Gaussian copula density with parameter $\rho$.

Under the causal DAG $A \rightarrow Y$ with unobserved confounding $U \rightarrow \{A, Y\}$, the model factorises as:

- $z_A = f_A(A)$ — monotone map from treatment to latent noise (no conditioning; $A$ has no observed parents)
- $z_Y = f_{Y \mid A}(Y;\, A)$ — monotone map from outcome to latent noise, conditioned on $A$

The joint latent vector is assumed to follow:

$$\begin{pmatrix} z_A \\ z_Y \end{pmatrix} \sim \mathcal{N}\!\left(0,\; \Sigma_\rho\right), \qquad \Sigma_\rho = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$$

After the causal effect of $A$ on $Y$ has been removed by $f_{Y \mid A}$, the remaining correlation $\rho$ between $z_A$ and $z_Y$ is entirely due to $U$.

---

## 4. How ρ-GNF Works in Depth

*(These details live in the backup slides, not the main presentation.)*

### 4.1 Architecture

**DAGConditioner** — computes an embedding $h_i$ for each variable $i$ using only its DAG parents:

$$h_i = \text{MLP}\!\left(M_{i,:} \odot x\right), \qquad M_{ij} = \mathbf{1}[j \in \text{pa}(i)]$$

With $A \rightarrow Y$: $h_A = \text{MLP}(\mathbf{0})$ (constant) and $h_Y = \text{MLP}(A)$.

**MonotonicNormalizer (UMNN)** — maps each variable to its latent noise using an unconstrained monotone neural network conditioned on the embedding:

$$z_i = \int_0^{x_i} g_\phi(t;\; h_i)\;\mathrm{d}t + b_i, \qquad g_\phi > 0$$

$$\log\left|\frac{\partial z_i}{\partial x_i}\right| = \log g_\phi(x_i;\; h_i)$$

### 4.2 Training Objective

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{n=1}^N \Bigl[\log p_{\mathcal{N}(0,\Sigma_\rho)}(z_A^{(n)}, z_Y^{(n)}) + \log\bigl|\det J_f(x^{(n)})\bigr|\Bigr]$$

The parameter $\rho$ is **not learned** — it is the sensitivity hyperparameter fixed per run. Only the flow weights $\theta$ are optimised.

### 4.3 ATE Estimation

$$\widehat{\text{ATE}}(\rho) = \frac{1}{S} \sum_{s=1}^{S} \Bigl[\bigl[f^{-1}(\mathbf{z}^{(s)},\; z_A{=}1)\bigr]_Y - \bigl[f^{-1}(\mathbf{z}^{(s)},\; z_A{=}0)\bigr]_Y\Bigr]$$

$$\text{where} \quad (z_A^{(s)},\, z_Y^{(s)}) \;\overset{\text{iid}}{\sim}\; \mathcal{N}(0, \Sigma_\rho), \quad z_A \text{ then pinned to } a \in \{0, 1\}$$

Do-intervention in the latent space is propagated through $f^{-1}$. The same paired $z_Y$ draws are used for both treatment arms (implicit variance reduction).

### 4.4 Binary Treatment (Note)

For binary treatment ($A \in \{0, 1\}$), the normalizing flow is applied after **dequantization**: small Gaussian noise is added to the binary values during training, making the density continuous. The do-operation then sets $A = 0$ or $A = 1$ directly in the data space. No propensity score is modelled.

---

## 5. Experiments

### 5.1 Setting 0: Gaussian Copula (Sanity Check — Assumption Satisfied)

**DGP:**

$$\begin{pmatrix}e_A \\ e_Y\end{pmatrix} \sim \mathcal{N}\!\left(0,\; \begin{pmatrix}1 & \rho^* \\ \rho^* & 1\end{pmatrix}\right), \quad A = e_A, \quad Y = \alpha A + e_Y$$

Parameters: $\alpha = 0.2$, $\rho^* = 0.30$. $n = 5{,}000$, 5 seeds.

The model is **correctly specified**: the Gaussian copula assumption matches the true DGP. The experiment asks whether the model recovers the correct ATE and whether the minimum RMSE coincides with the true $\rho^*$.

**ρ-curve results** (selected rows from 20 assumed ρ values):

| Assumed $\rho$ | Mean ATE est | RMSE |
|:---:|:---:|:---:|
| $0.156$ | $0.343$ | $0.149$ |
| $\mathbf{0.261}$ | $\mathbf{0.228}$ | $\mathbf{0.044}$ ← min |
| $0.365$ | $0.109$ | $0.096$ |
| $0.990$ | $-0.783$ | $1.012$ |

**Key findings:**
- The ATE estimate is **extremely sensitive** to $\rho$: across the full sweep the mean estimate ranges from $+2.0$ to $-0.8$.
- Minimum RMSE at $\rho \approx 0.261$, close to the true $\rho^* = 0.30$. Bias at the optimum is $< 0.03$.
- Confirms: when the Gaussian copula assumption is satisfied, $\rho$-GNF is well-calibrated and the ρ-curve is a valid diagnostic.

### 5.2 Setting 1: Clayton Copula (Misspecification)

**DGP:** Replace the Gaussian copula with a **Clayton copula** (lower-tail dependence only):

$$C_\theta(u_A, u_Y) = \left(u_A^{-\theta} + u_Y^{-\theta} - 1\right)^{-1/\theta}, \quad \lambda_L = 2^{-1/\theta}$$

$$A = \Phi^{-1}(u_A), \quad Y = \alpha A + \Phi^{-1}(u_Y), \quad (u_A, u_Y) \sim \text{Clayton}(\theta = 2)$$

Parameters: $\alpha = 0.2$, Kendall's $\tau = 0.50$, $\rho_{\text{ref}} = \sin(\pi \tau / 2) \approx 0.707$. The model still uses a Gaussian copula.

**ρ-curve results** (10 assumed ρ values):

| Assumed $\rho$ | Mean ATE est | RMSE |
|:---:|:---:|:---:|
| $\rho_{\text{ref}} = 0.707$ | $-0.038$ | $0.240$ |
| $\mathbf{\rho_{\text{opt}} = 0.550}$ | $\mathbf{0.216}$ | $\mathbf{0.053}$ ← min |

**Key findings:**
- The Gaussian-equivalent $\rho_{\text{ref}} = 0.707$ does **not** minimise RMSE. The optimal assumed $\rho$ is $\approx 0.55$, noticeably below $\rho_{\text{ref}}$.
- RMSE at $\rho_{\text{ref}}$ is $4.5\times$ higher than at $\rho_{\text{opt}}$.
- The Gaussian flow compensates for the Clayton tail structure by shifting $\rho$ — but it cannot reproduce the asymmetric tail dependence ($\lambda_L > 0$ for Clayton, $\lambda_L = 0$ for Gaussian).

---

## 6. Limitations

Two structural limitations of the original ρ-GNF setup are directly motivated by the experimental evidence above.

### 6.1 Limitation 1: Gaussian Copula Is a Forced Assumption

The latent prior is always:

$$\begin{pmatrix}z_A \\ z_Y\end{pmatrix} \sim \mathcal{N}(0, \Sigma_\rho)$$

regardless of the true DGP. This means:

- **Tail dependence** cannot be represented. The Gaussian copula has zero upper and lower tail dependence for any $\rho < 1$, while Clayton has $\lambda_L = 2^{-1/\theta} > 0$. No choice of $\rho$ corrects this.
- **Optimal $\rho$ shifts** away from the Kendall-matched reference: the model finds the $\rho$ that minimises a misspecified likelihood, not the copula-correct one.
- The ρ-curve loses its clean interpretation: the true ATE no longer lies on the curve at the "right" $\rho$ — it is displaced by the copula shape mismatch.

### 6.2 Limitation 2: ρ Is Blind to Observed Covariates

$\rho$ is a single global constant — **the same for every observation**, even when the confounding strength clearly varies across subgroups defined by observed covariate $X$.

**Concrete setting (Setting 2 DGP):**

$$\theta(X) = 0.5 + 3.5\,X, \quad X \sim U[0,1]$$

- $X \approx 0$: weak confounding ($\theta \approx 0.5$, $\tau \approx 0.20$)
- $X \approx 1$: strong confounding ($\theta \approx 4.0$, $\tau \approx 0.67$)

A single global $\rho$ will be too strong for low-$X$ observations and too weak for high-$X$ observations simultaneously. The original ρ-GNF has no mechanism to use $X$ when estimating $\hat\tau$.

---

## 7. Extension: Context Conditioning

### 7.1 The Implementation Change (Part 0a)

The `DAGConditioner.forward()` method accepted a `context` argument but silently discarded it. The fix concatenates the context onto the masked embedding before the MLP:

**Before:**
```python
def forward(self, x, context=None):
    e = (x.unsqueeze(1) * self.A.unsqueeze(0)).reshape(B * d, d)
    return self.embedding_net(e).view(B, d, -1)
```

**After:**
```python
def forward(self, x, context=None):
    e = (x.unsqueeze(1) * self.A.unsqueeze(0)).reshape(B * d, d)
    if context is not None:
        ctx = context.unsqueeze(1).expand(-1, d, -1).reshape(B * d, -1)
        e = torch.cat([e, ctx], dim=-1)
    return self.embedding_net(e).view(B, d, -1)
```

The `DAGMLP` first linear layer is widened by `cond_in` at construction time. No other changes are needed: the context flows through the existing `context=` arguments throughout `FCNormalizingFlow` and its inversion.

### 7.2 The Two Model Variants

**V1 — no context (baseline):**
$$h_i = \text{MLP}(M_{i,:} \odot x)$$
Single global $\rho$. Cannot distinguish high-$X$ from low-$X$ samples.

**V2 — context X:**
$$h_i = \text{MLP}(M_{i,:} \odot x \;\|\; X)$$
Embedding MLP receives $X$. The flow transformations become X-dependent.

### 7.3 Context-Conditioned ATE Estimator

For V2, the ATE marginalises over the empirical distribution of $X$:

$$\widehat{\text{ATE}}_{\text{V2}}(\rho) = \frac{1}{S}\sum_{s=1}^{S}\Bigl[\bigl[f^{-1}(\mathbf{z}^{(s)},\; z_A{=}1,\, X_s)\bigr]_Y - \bigl[f^{-1}(\mathbf{z}^{(s)},\; z_A{=}0,\, X_s)\bigr]_Y\Bigr]$$

where $X_s$ is drawn with replacement from the observed $X$ values.

### 7.4 What Context Can and Cannot Fix

| Effect | V1 | V2 |
|---|---|---|
| X-conditional marginal distributions of $(A,Y)$ | ✗ Cannot adapt | ✓ Flow transformations are X-conditional |
| Copula **shape** mismatch (Clayton vs Gaussian) | ✗ | ✗ Still globally misspecified |
| Copula **strength** varying with $X$ ($\rho = \rho(X)$) | ✗ | ✗ Global $\rho$ fixed (needs Part 0b) |

V2 strictly improves V1 on density fit. It does not change the Gaussian copula in the latent space.

### 7.5 Experiment Design — Setting 2

Setting 2 sweeps 10 values of $\rho$ from $-0.99$ to $+0.99$, running V1 and V2 in parallel for 3 seeds each ($n = 5{,}000$, $\alpha = 0.2$). The comparison plots:
- ρ-curves: mean ATE $\pm$ std for each variant
- RMSE vs assumed $\rho$ for each variant

**Status: results pending.**

---

## 8. Conclusions

### 8.1 Summary of Findings

| Setting | Model | Key result |
|---|---|---|
| Gaussian copula (Setting 0) | V1 | Well-calibrated at true $\rho^*$. RMSE = 0.044. ρ-curve is steep and informative. |
| Clayton copula (Setting 1) | V1 | RMSE at $\rho_{\text{ref}}$ is $4.5\times$ higher than at $\rho_{\text{opt}}$. Optimal ρ shifts to 0.55. |
| Heterogeneous Clayton (Setting 2) | V1 vs V2 | Pending — expected V2 RMSE $<$ V1 RMSE at the same $\rho$. |

The ρ-GNF framework correctly solves the sensitivity analysis problem **when the Gaussian copula assumption holds**. The ρ-curve is a valid, informative diagnostic: the true ATE lies on it if and only if the true $\rho$ is known and the copula is correctly specified.

When the assumption is violated (wrong copula shape, heterogeneous confounding), the model degrades in two ways:

1. **Curve misalignment:** The true ATE no longer lies on the ρ-curve at the correct $\rho$.
2. **Directional bias:** The Clayton copula's lower-tail dependence induces a systematic shift in ATE estimates.

### 8.2 Future Work

**Part 0b — Conditional Copula**

Replace the fixed $\Sigma_\rho$ with a small neural network $\rho_\text{net}(X) \to \rho(X) \in (-1, +1)$:

$$z_{A,i},\, z_{Y,i} \sim \mathcal{N}\!\left(0,\; \Sigma_{\rho(X_i)}\right), \qquad \rho(X_i) = \tanh\!\left(\text{MLP}(X_i)\right)$$

This is V3. It directly allows the confounding strength to vary with $X$, addressing Limitation 2.

**Beyond Gaussian Copulas**

Replace the MVN prior on $(z_A, z_Y)$ with a flexible normalizing flow prior, separating the causal DAG structure (outer flow) from the confounding structure (inner prior) without fixing a parametric copula family. This addresses Limitation 1.

---

## Appendix: Technical Details

### A.1 UMNN

$$z = \int_0^x g_\phi(t;\, h) \,\mathrm{d}t + b, \qquad g_\phi > 0 \text{ (ELU + 1)}$$

Inverse $x = \text{UMNN}^{-1}(z;\, h)$ computed by bisection on the monotone function.

### A.2 Copula Parameter Reference

| Copula | Parameter | Kendall's $\tau$ | Tail dependence |
|---|---|---|---|
| Gaussian | $\rho$ | $\frac{2}{\pi}\arcsin(\rho)$ | $\lambda_U = \lambda_L = 0$ |
| Clayton | $\theta > 0$ | $\frac{\theta}{\theta + 2}$ | $\lambda_L = 2^{-1/\theta}$, $\lambda_U = 0$ |
| t-copula | $(\rho, \nu)$ | $\frac{2}{\pi}\arcsin(\rho)$ | $\lambda_U = \lambda_L > 0$ |

Kendall's $\tau$ matching — Gaussian $\rho$ equivalent to Clayton $\theta$:

$$\rho_{\text{ref}} = \sin\!\left(\frac{\pi}{2} \cdot \frac{\theta}{\theta + 2}\right) \approx 0.707 \quad \text{at } \theta = 2$$
