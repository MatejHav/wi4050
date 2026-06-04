# Additional Experiments

Six targeted experiments to strengthen the presentation's argument.

---

## 1. OLS Baseline on Every ρ-curve

**What.** Add a horizontal dashed line to the Setting 0 and Setting 1 ρ-curve plots at the naïve OLS estimate (i.e. the ATE you get at ρ = 0 without any adjustment).

**Why.** Right now the audience sees a curve but has no anchor for how bad the unadjusted estimate is. Seeing "OLS gives 0.41, truth is 0.20" makes the severity of confounding tangible and motivates the whole method in one visual.

**How.** Run `setting0_baseline.py` (which already exists) to extract the OLS/IPW ATE, then add a `plt.axhline` to the ρ-curve plot at that value, labelled "naïve OLS".

---

## 2. ATE Zero-Crossing Annotation

**What.** On each ρ-curve, annotate the value of ρ at which the estimated ATE crosses zero (the tipping point: the confounding strength needed to explain the effect away entirely).

**Why.** This is the single most practically useful output of a sensitivity analysis. Currently the plots show where RMSE is minimised, not where conclusions flip. A reviewer or audience member asking "is this effect robust?" needs to see the tipping point, not the minimum RMSE.

**How.** After fitting the ρ-curve, find the zero-crossing by linear interpolation between the two bracketing ρ values. Add a vertical annotation line and label "effect nullified at ρ ≈ X". One function, reusable across all settings.

---

## 3. V2 vs V1 Statistical Significance Test

**What.** Rerun Setting 2 with 20 seeds instead of 3. For each assumed ρ, perform a paired t-test (or bootstrap CI) on the RMSE difference V2 − V1.

**Why.** The current 3-seed comparison shows almost no difference. Either the effect is real and tiny (which should be stated honestly), or it is noise (which should be stated even more honestly). With 3 seeds and visually overlapping confidence bands, the claim "V2 has lower variance" is not credible. If the difference is not significant at any ρ, that is a strong result: it says conditioning X in the flow gives no measurable benefit when the copula is still misspecified.

**How.** Modify `setting2_context.py` to use `n_seeds=20`. Add a panel to `comparison.png` showing the RMSE difference with a bootstrap 95% CI shaded band and a horizontal line at 0.

---

## 4. V3 Implementation and Setting 2 Comparison

**What.** Implement V3: replace the fixed scalar ρ with `rho_net(X) = tanh(MLP(X))`, draw per-observation latent noise from `N(0, Σ_{ρ(X_i)})`, and run on the heterogeneous Clayton DGP from Setting 2.

**Why.** The presentation proposes V3 as the natural next step but shows no result. Without it, the "Further Extension" slide is a hypothesis, not evidence. Even a partial result (e.g. V3 RMSE at the optimal ρ vs V1/V2) is far stronger than a blank slide. If V3 also fails to close the gap, that becomes the conclusion: the Gaussian copula shape assumption is the binding constraint, not the scalar ρ.

**How.**
- In `DAGConditioner.__init__`, add a small `rho_net = nn.Sequential(nn.Linear(cond_dim, 16), nn.Tanh(), nn.Linear(16, 1), nn.Tanh())`.
- Pass per-sample ρ values when constructing the log-likelihood in `FCNormalizingFlow`.
- Re-run the Setting 2 sweep and add V3 to the comparison plot.

---

## 5. Misspecification Severity vs Clayton θ

**What.** Run Setting 1 for θ ∈ {0.5, 1, 2, 4, 8} and plot (a) the ratio RMSE(ρ_ref) / RMSE(ρ_opt) and (b) the gap |ρ_opt − ρ_ref| as a function of θ.

**Why.** Right now Setting 1 uses a single θ = 2. The audience cannot tell whether the result is a mild problem or a severe one, or how quickly it degrades. Showing the degradation curve as a function of tail dependence strength (λ_L = 2^{−1/θ}) gives the method a failure mode that can be characterised quantitatively, which is much more informative than one data point.

**How.** Parameterise `setting1_rho_sweep.py` over θ. Store RMSE arrays per θ, then produce a two-panel figure: left = RMSE ratio vs θ, right = optimal-ρ shift vs θ. Expected shape: monotone degradation as θ increases (stronger tail dependence).

---

## 6. Comparison to E-value

**What.** For the Setting 0 result, compute the E-value (VanderWeele & Ding 2017) for the naïve OLS estimate and compare it to the ρ-GNF tipping-point ρ from Experiment 2 above.

**Why.** This directly answers the most likely audience question: "Why not just use E-values?" The E-value is nonparametric and requires no model, so it is both simpler and harder to dismiss. If ρ-GNF's tipping-point translates to a comparable E-value, the method adds complexity with no gain. If it gives a tighter or more informative bound (e.g. because the flow captures the nonlinear outcome model), that is the method's actual value-add, and should be the headline claim.

**How.** The E-value for a risk ratio RR is `RR + sqrt(RR*(RR-1))`. For a continuous outcome, use the approximation via the standardised effect size (Cinelli & Hazlett 2020 partial R² approach is an alternative). Compare the implied confounding strength in both frameworks.
