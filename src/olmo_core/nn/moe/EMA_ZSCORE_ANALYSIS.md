# MoE Router EMA Z-Score Normalization — Holt's Redesign

## Summary

The previous `MoERouter.ema_zscore_normalize` path with Holt's trend enabled had a structural failure mode that caused router runaway around step ~1087 on a 1B-param MoE run. The failure is not a hyperparameter tuning issue; it is an algebraic consequence of running two independent Holt's linear-trend predictors on `E[X]` and `E[X²]` and reconstructing variance as `σ² = E[X²] − E[X]²`.

This document records the diagnosis and the redesign. The fix:

1. Track `Var(logit)` directly, not `E[X²]`, using a sample-mean-based observation `var_obs = sq_sum/count − μ_obs²`.
2. Use Holt's linear-trend smoothing (undamped) for the mean only.
3. Use plain EMA (no trend) for the variance.

---

## 1. The failure

### Symptom

- `σ̂` (the forecast standard deviation used to z-normalize router logits) was clamped to the safety floor `1e-2` (i.e. `Var̂ = 1e-4`) starting at step **1087** for one expert.
- The cascade spread to 13 experts by step 1150, 20 by step 1200, 22 by step 1300.
- 63 of 64 experts eventually touched the floor at some point; one expert spent 59% of post-activation steps pinned there.
- Once an expert's `σ̂` floors, the z-normalized logit `(logit − μ̂) / σ̂` amplifies by ~100×, which blows up softmax/sigmoid routing for that expert and shocks the system.

### Root cause (the cross-term cancellation)

Reconstructing variance from two independently-smoothed moments:

```
var_hat = (level_sq + trend_sq) − (level_μ + trend_μ)²
        = (level_sq − level_μ²) + (trend_sq − 2·level_μ·trend_μ − trend_μ²)
        = var_level            +  trend_contribution_to_variance
```

Under sustained drift in `μ`, `level_μ` drifts away from zero and `trend_μ` becomes non-zero. The cross-term `−2 · level_μ · trend_μ` accumulates and cancels `var_level`. The trend predictor on `E[X²]` cannot compensate, because its true target `dE[X²]/dt = 2·μ·dμ/dt` is linear in `μ` and therefore grows over time — a dynamic that a linear-slope predictor cannot fit. It systematically lags, and the cross-term wins.

Direct measurement on the first expert to floor (expert 37 at step 1087):

| step | var_level | trend_sq | 2·level_μ·trend_μ | trend contrib | var_hat |
|---|---|---|---|---|---|
| 1072 | 0.92 | +0.04 | +0.16 | +0.04 − 0.16 = −0.12 | 0.64 |
| 1085 | 0.56 | +0.09 | +0.17 | −0.09 | 0.058 |
| 1087 | 0.056 | +0.10 | +0.16 | −0.056 | **1e-4 (clamp)** |

`var_level` shrank from 0.92 to 0.056 as the mean drift accelerated, and the cross-term then finished the job. The floor clamped; routing destabilized; expert variance oscillated between 0.01 and 24 for the remainder of the run.

### Why it only appeared after step ~1000

The code includes a 1000-step plain-EMA warmup (`ema_zscore_trend_warmup`, default 1000) that holds trend buffers at zero and runs plain EMA. At step 1000 the warmup ends, the trend buffers start accumulating, and the cross-term begins eating `var_level`. The 87 additional steps between 1000 and 1087 match the time required for the bias to compound until it equals the lowest-variance expert's `var_level`.

---

## 2. Why plain EMA alone is not sufficient either

Removing Holt's and running plain EMA on both `μ` and `Var` avoids the cancellation but introduces a second failure mode: winner-takes-all routing via the 100-step lag of the mean estimator.

Measured on a 3,290-step plain-EMA-only training run (same model, same seed family):

- 15 of 64 experts have monotone mean drift (0–1 sign flips across halves); correlation of drift slope across halves is **0.929**.
- Median R² of a single global line fit to `μ(t)` per expert is **0.975**. Drift is approximately linear.
- At α = 0.99, the steady-state plain-EMA lag on `μ` is `99 · slope` per expert.
- Translated to normalized-logit bias (per-expert forecast error divided by the expert's own σ): median 0.09σ, max **0.44σ**.
- Softmax favoritism factor from this bias (simulated on 200k synthetic tokens): max **1.55×**, max/min ratio **2.11×**.

A sustained 1.55× per-step share advantage compounds through gradient feedback into the linearly-growing `tokens percentage` trajectory observed in the plain-EMA run. Plain EMA is mechanically stable (no floor hits, no numerical issues) but cannot prevent the router from collapsing toward a few experts.

---

## 3. Why `μ` gets Holt's but `Var` does not

The asymmetry is not arbitrary — it follows from how each bias propagates through the softmax.

### Drift signal

| Quantity | median drift SNR | 95th percentile |
|---|---|---|
| `μ(t)` | 21.4 | 90.5 |
| `Var(t)` | 6.6 | 23.2 |

Both have real drift, but `Var` drift is roughly 3× weaker and the series is more reversal-prone (4 of 64 experts monotone vs. 15 for `μ`).

### Softmax propagation: different orders of effect

A bias `δ` in the mean forecast (in σ-units) shifts the z-normalized logit directly by `−δ` → softmax share multiplier `exp(−δ)`. Linear in `δ`.

A bias `ε` in the std forecast (as a relative factor `σ̂ = (1+ε)·σ_true`) shifts the variance of the z-normalized logit by `≈ (1+ε)⁻²` → softmax share multiplier `exp(Var[z]/2 · [(1+ε)⁻² − 1])`. This is effectively linear in `ε` but scaled by Jensen's convexity on `exp`; at typical values it is **~12× weaker per unit of relative bias** than the mean-bias effect.

Simulation on 200k synthetic tokens, 64 experts, using bias magnitudes matching plain-EMA lag on the actual data:

| Case | max favoritism | max/min ratio | log-ratio contribution |
|---|---|---|---|
| μ̂ bias only (0.44σ) | 1.50× | 2.11× | 0.745 |
| σ̂ bias only (±3.5%) | 1.03× | 1.06× | 0.061 |
| Combined | 1.51× | 2.13× | 0.757 |

`σ̂` bias accounts for ~8% of total routing asymmetry at the plain-EMA lag level. Also, unlike μ̂ bias, it does not feed back through the gradient: skewed sharpness does not bias the *direction* the expert's logit drifts, only how peaked its score distribution is.

### Overshoot risk

Plain EMA has bounded lag; it cannot drive `σ̂²` below zero. Holt's with trend extrapolation *can* — a transient downward shock in `Var` plus a positive slope memory can produce `level + trend < 0` and hit the floor. Since `Var` is bounded below by 0 and the drift signal is weaker, this is net-negative.

### Conclusion

- `μ` pays 2.11× routing imbalance for plain EMA's 99-step lag; Holt's takes it to 1.00×.
- `Var` pays 1.06× imbalance for plain EMA's ~7% lag; Holt's would take it to ~1.00× but introduces overshoot-past-zero risk.
- Plain EMA on `Var` is both simpler (one fewer buffer, no trend hyperparameter) and strictly safer.

---

## 4. Why undamped (`φ = 1`), not damped

The textbook automatic-forecasting wisdom (Gardner-McKenzie 1985; Hyndman et al.) says damped Holt's is the winner across the M3/M4 competitions. Two reasons that advantage does not transfer here:

1. **h = 1 forecast horizon.** Damping's canonical value is protecting multi-step forecasts from phantom-slope extrapolation. At `h = 1` the extrapolation is a single step worth of trend; there is less to protect against.
2. **Measured drift is approximately constant.** Linear R² of 0.975, same-sign half-to-half slope in 50/64 experts. In this regime, undamped Holt's has zero steady-state bias while damping introduces a bias of `−(1−φ)/(1−α) · slope` per expert.

Simulation on the real `y_μ` observations from the 3k-step plain-EMA run (pre-seeded `b_0` from the 50 pre-warmup steps, forecast error after a 100-step transient):

| Method | median MAE | median \|bias\| | max \|bias\| |
|---|---|---|---|
| Plain EMA | 0.094 | 0.040 | 0.518 |
| Holt's, φ = 1.00 | **0.042** | **0.0003** | **0.003** |
| Holt's, φ = 0.98 | 0.042 | 0.0012 | 0.013 |
| Holt's, φ = 0.95 | 0.041 | 0.0024 | 0.028 |

All Holt's variants are ~2× better than plain EMA in MAE. Undamped is 10× better in max bias than `φ = 0.98` with equivalent MAE. The "reversals cost overshoot" argument I ran in the canonical stimulus simulation did not materialize on the real data because real reversals are rare and shallow — undamped does not accumulate phantom slope in practice.

Softmax favoritism on the actual measured biases:
- Plain EMA: [0.68×, 1.17×]
- Holt's φ = 1.00: [1.00×, 1.00×]
- Holt's φ = 0.98: [0.99×, 1.00×]

Undamped wins.

---

## 5. Why `var_obs` uses the sample mean, not the forecast mean

A first draft of the redesign computed the variance observation as
`var_obs = global_sq − 2·μ̂·global_mean + μ̂²`, which is `(1/n) Σ (x_i − μ̂)²` where `μ̂` is the forecast.

That is biased: `E[(X − μ̂)²] = Var(X) + (μ_true − μ̂)²`. The second term is the squared forecast error, which is positive whenever Holt's has not perfectly caught the drift. Smoothing a `var_obs` that includes this term feeds phantom variance into `σ̂`, pushing the normalizer too large and re-introducing a correlated bias.

The sample-mean-based observation `var_obs = global_sq − μ_obs²` is `(1/n) Σ (x_i − x̄)²`, which is an unbiased-up-to-`(n−1)/n` estimator of `Var(X)` and completely decouples from `μ̂`. At our batch sizes the Bessel correction is irrelevant.

Both versions use exactly the same accumulators (`sum`, `sq_sum`, `count`) — the difference is one line of arithmetic.

---

## 6. Final design

**Config** (`MoERouterConfig`):

- `ema_zscore_normalize: bool = False` — enables the whole path.
- `ema_zscore_alpha: float = 0.99` — EMA decay, used for both `μ`-level and `Var`-level.
- `ema_zscore_trend: bool = False` — enables undamped Holt's on **μ only**.
- `ema_zscore_trend_beta: float = 0.9` — Holt's β (applied to μ's trend).
- `ema_zscore_trend_warmup: int = 1000` — plain-EMA-on-μ phase before Holt's kicks in. Bias correction makes the transition continuous.

**State (buffers)**:

- `_ema_mean` — plain-EMA / Holt's level for μ.
- `_ema_var` — plain-EMA level for `Var(logit)`. Replaces the old `_ema_sq` (which tracked `E[X²]`).
- `_ema_mean_trend` — Holt's trend for μ. Present only when `ema_zscore_trend = True`.
- `_ema_step_count` — for Adam-style bias correction `bc = 1 − α^t`.

**Observations** (computed once per optimizer step after SUM-and-COUNT all-reduce):

```
μ_obs    = sum / count
var_obs  = sq_sum / count − μ_obs²   (sample-mean based, unbiased of μ̂)
```

**Updates** (after warmup step count):

```
# μ: Holt's (if ema_zscore_trend and step ≥ warmup), else plain EMA
if use_trend:
    level_μ_old = level_μ.clone()
    level_μ  ← α · (level_μ + trend_μ) + (1 − α) · μ_obs
    trend_μ  ← β · trend_μ + (1 − β) · (level_μ − level_μ_old)
else:
    level_μ  ← α · level_μ + (1 − α) · μ_obs

# Var: plain EMA, always
level_var ← α · level_var + (1 − α) · var_obs
```

**Forecast** (read in forward):

```
bc = 1 − α^step

if use_trend:
    μ̂  = (level_μ + trend_μ) / bc
else:
    μ̂  = level_μ / bc

σ̂² = max(level_var / bc, 1e-4)
σ̂  = sqrt(σ̂²)

z = (logits − μ̂) / σ̂
```

The `1e-4` clamp stays in as a belt-and-suspenders safety — under the new formulation it should never trigger on drift (the cancellation mechanism is gone), so if it does, it signals a genuine variance collapse worth logging.

---

## 7. Validation against failures

| Failure mode | Old code | New code |
|---|---|---|
| Cross-term cancellation → floor clamp | Hits floor at step 1087, cascades to 63/64 experts | Impossible by construction: `Var` is tracked directly; no `E[X²] − μ²` reconstruction. |
| `σ̂` under-estimate during μ drift | ~35% under-estimate at expert 37 step 1086 | Zero steady-state bias on μ; `Var` observation is sample-mean-based, no coupling to μ̂ error. |
| Winner-takes-all from μ̂ lag | Under plain EMA: 2.11× favoritism | Holt's on μ (undamped) drops this to 1.00× per-step. |
| Overshoot-past-zero on `Var` | Hit by the old formulation via clamp | `Var` uses plain EMA → monotone decay, bounded by observations. Cannot project below `min(var_obs)`. |
| Startup transient | Holt's trend buffer init at 0 + no warmup = biased first ~100 steps | 1000-step plain-EMA warmup on μ, then seamless Holt's switch-on via bc continuity. |

---

## 8. Checkpoint compatibility

Renaming `_ema_sq` → `_ema_var` and dropping `_ema_sq_trend` is a breaking change. Old checkpoints with `ema_zscore_normalize = True` cannot be loaded directly into the new code. This is acceptable because:

- The only Holt's-enabled checkpoints in existence are from the broken run; loading them is not useful.
- Plain-EMA-only checkpoints (`ema_zscore_trend = False`) had `_ema_sq` tracking `E[X²]`, which does not transfer correctly to `_ema_var` (which tracks `Var`). A migration would need to subtract `_ema_mean²` from the loaded `_ema_sq`, possibly divide by bc. Not worth the code.

Fresh initialization is expected for this feature going forward.
