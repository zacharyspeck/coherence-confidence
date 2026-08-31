# coherence-confidence — analysis

- run: `mock-coherence_driven__107ab946de39__2f392d0ec884fc6a`
- model: `mock:coherence_driven` (revision `None`)
- template hash: `2f392d0ec884fc6a`
- options: `['Yes', 'No', 'Unsure']`
- scored: 80 items in 20 families
- timestamp: 2026-08-31T17:35:08+00:00

> **SYNTHETIC RUN.** These are fixtures, not measurements.


## 1. PRIMARY ENDPOINT — AUC(coherent) vs AUC(diverse)

> AUC(coherent) - AUC(diverse) on the FULL set. Koriat's consensuality principle predicts NEGATIVE: coherence raises confidence without raising accuracy, so the confidence-accuracy relationship degrades where the evidence agrees with itself. AUC(coherent) below 0.5 is the crossover.

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5000** | [+0.5000, +0.5000] | 400 | 0 | 400 | — |
| diverse | **0.5000** | [+0.5000, +0.5000] | 400 | 0 | 400 | — |

**AUC(coherent) − AUC(diverse) = +0.0000**  95% CI (family) [+0.0000, +0.0000]
— sign is exactly zero — neither direction; the interval spans zero, so the direction is not resolved.

## 2. The same endpoint, confound-controlled

> The same gap on the MATCHED-MECHANISM subset, where both conditions are falsified by scope_mismatch, so flaw mechanism cannot differ between them and only coherence does (D-024).

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5000** | [+0.5000, +0.5000] | 100 | 0 | 100 | — |
| diverse | **0.5000** | [+0.5000, +0.5000] | 100 | 0 | 100 | — |

**AUC(coherent) − AUC(diverse) = +0.0000**  95% CI (family) [+0.0000, +0.0000]
— sign is exactly zero — neither direction; the interval spans zero, so the direction is not resolved.

Both conditions here are falsified by the *same* mechanism, so a gap cannot be attributed to one cell's flaws being a different kind of thing from the other's. This is the number to quote when asked whether the coherent-false items are simply easier to catch.

## 3. Per-mechanism breakdown

| cell | stated_confound | broken_chronology | scope_mismatch |
|---|---|---|---|
| `coherent_false` | 10 | 0 | 10 |
| `diverse_false` | 0 | 10 | 10 |

### `stated_confound`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5000** | [+0.5000, +0.5000] | 100 | 0 | 100 | — |

### `broken_chronology`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| diverse | **0.5000** | [+0.5000, +0.5000] | 100 | 0 | 100 | — |

### `scope_mismatch`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5000** | [+0.5000, +0.5000] | 100 | 0 | 100 | — |
| diverse | **0.5000** | [+0.5000, +0.5000] | 100 | 0 | 100 | — |

**AUC(coherent) − AUC(diverse) = +0.0000**  95% CI (family) [+0.0000, +0.0000]
— sign is exactly zero — neither direction; the interval spans zero, so the direction is not resolved.

## 4. Salience, reported as a covariate

| cell | n with salience | mean | max |
|---|---|---|---|
| `coherent_false` | 0 | — | — |
| `diverse_false` | 0 | — | — |

_Logistic model not fitted: only 0 FALSE items carry a salience value; run the blind audit and scripts/apply_salience.py first_


## 5. Diagnostics (not endpoints)

> Cell means, the 2x2 effects and the ANOVA are diagnostics. They describe confidence level; the endpoint is the confidence-accuracy relationship, which is the AUC.

### Mean confidence per cell — P(yes) three-way

| cell | n | mean | 95% CI (item) |
|---|---|---|---|
| `coherent_true` | 20 | 0.8500 | [0.8500, 0.8500] |
| `coherent_false` | 20 | 0.8500 | [0.8500, 0.8500] |
| `diverse_true` | 20 | 0.4500 | [0.4500, 0.4500] |
| `diverse_false` | 20 | 0.4500 | [0.4500, 0.4500] |

### Mean confidence per cell — P(yes) two-way (yes vs no only)

| cell | mean | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.8901 | [0.8901, 0.8901] |
| `coherent_false` | 0.8901 | [0.8901, 0.8901] |
| `diverse_true` | 0.5389 | [0.5389, 0.5389] |
| `diverse_false` | 0.5389 | [0.5389, 0.5389] |

### Abstention rate per cell

Fraction of items where the third option has the highest probability.

| cell | rate | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.0000 | [0.0000, 0.0000] |
| `coherent_false` | 0.0000 | [0.0000, 0.0000] |
| `diverse_true` | 0.0000 | [0.0000, 0.0000] |
| `diverse_false` | 0.0000 | [0.0000, 0.0000] |


### 2x2 effects on mean confidence

> Read the family-clustered CI for these. The 2x2 is within-family, so clustering keeps the pairing; the item-level interval discards it. See DECISIONS.md D-010.

| effect | estimate | 95% CI (item) | **95% CI (family)** |
|---|---|---|---|
| main_effect_coherence | +0.4000 | [+0.4000, +0.4000] | [+0.4000, +0.4000] |
| main_effect_truth | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] |
| interaction | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] |
| coherence_effect_within_true | +0.4000 | [+0.4000, +0.4000] | [+0.4000, +0.4000] |
| coherence_effect_within_false | +0.4000 | [+0.4000, +0.4000] | [+0.4000, +0.4000] |

- **main_effect_coherence** — mean(coherent) - mean(diverse), collapsing over truth
- **main_effect_truth** — mean(true) - mean(false), collapsing over coherence
- **interaction** — (coh_true - coh_false) - (div_true - div_false)
- **coherence_effect_within_true** — D-004: the CLEAN coherence contrast. Both cells are flawless, so neither the flaw-type confound nor flaw salience can touch it.
- **coherence_effect_within_false** — D-004: mixes coherence with whatever differs between the mechanisms.

### Two-way ANOVA-style breakdown

| source | SS | df | MS | F | p | partial eta^2 |
|---|---|---|---|---|---|---|
| coherence | 3.20000 | 1 | 3.20000 | 394614561248646673795702818078720.000 | 0.00000 | 1.0000 |
| truth | 0.00000 | 1 | 0.00000 | 0.000 | 1.00000 | 0.0000 |
| coherence x truth | 0.00000 | 1 | 0.00000 | 304.000 | 0.00000 | 0.8000 |
| residual | 0.00000 | 76 | 0.00000 |  |  |  |

> F-tests assume independent observations. Items come in families of 4 sharing a scenario, so they are not independent; treat these as a variance breakdown and read the bootstrap CIs on the effect estimates for inference.

### Coverage

- option mass covered: mean 0.6000, min 0.6000, max 0.6000
- low coverage means the three options hold little of the next-token distribution and the renormalized numbers are ratios of small numbers.
