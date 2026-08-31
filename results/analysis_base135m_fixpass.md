# coherence-confidence — analysis

- run: `HuggingFaceTB_SmolLM2-135M__107ab946de39__efdbe3c2fc149d59`
- model: `HuggingFaceTB/SmolLM2-135M` (revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`)
- template hash: `efdbe3c2fc149d59`
- options: `['Yes', 'No', 'Maybe']`
- scored: 80 items in 20 families
- timestamp: 2026-08-31T18:05:32+00:00

## 1. PRIMARY ENDPOINT — AUC(coherent) vs AUC(diverse)

> AUC(coherent) - AUC(diverse) on the FULL set. Koriat's consensuality principle predicts NEGATIVE: coherence raises confidence without raising accuracy, so the confidence-accuracy relationship degrades where the evidence agrees with itself. AUC(coherent) below 0.5 is the crossover.

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5025** | [+0.4250, +0.5775] | 400 | 201 | 0 | 2.83 |
| diverse | **0.4825** | [+0.4175, +0.5475] | 400 | 193 | 0 | 2.65 |

**AUC(coherent) − AUC(diverse) = +0.0200**  95% CI (family) [-0.1026, +0.1400]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.
— **AUC(diverse) = 0.4825 is below 0.5**: confidence runs backwards against truth in that condition. That is the crossover, not merely a smaller effect.
— salience gap over the same items: +0.18 of 5 — the coherent flaws were louder, which pushes this AUC gap upward and therefore *against* the hypothesis - the test is conservative here.

## 2. The same endpoint, confound-controlled

> The same gap on the MATCHED-MECHANISM subset, where both conditions are falsified by scope_mismatch, so flaw mechanism cannot differ between them and only coherence does (D-024).

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5200** | [+0.3749, +0.6800] | 100 | 52 | 0 | 2.70 |
| diverse | **0.4200** | [+0.2778, +0.5312] | 100 | 42 | 0 | 2.75 |

**AUC(coherent) − AUC(diverse) = +0.1000**  95% CI (family) [-0.1389, +0.3600]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.
— **AUC(diverse) = 0.4200 is below 0.5**: confidence runs backwards against truth in that condition. That is the crossover, not merely a smaller effect.
— salience gap over the same items: -0.05 of 5 — effectively equalized, so this AUC gap is not a salience artifact in either direction.

Both conditions here are falsified by the *same* mechanism, so a gap cannot be attributed to one cell's flaws being a different kind of thing from the other's. This is the number to quote when asked whether the coherent-false items are simply easier to catch.

## 3. Per-mechanism breakdown

| cell | stated_confound | broken_chronology | scope_mismatch |
|---|---|---|---|
| `coherent_false` | 10 | 0 | 10 |
| `diverse_false` | 0 | 10 | 10 |

### `stated_confound`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.4900** | [+0.3827, +0.5833] | 100 | 49 | 0 | 2.95 |

### `broken_chronology`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| diverse | **0.5300** | [+0.4200, +0.6667] | 100 | 53 | 0 | 2.55 |

### `scope_mismatch`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5200** | [+0.3749, +0.6800] | 100 | 52 | 0 | 2.70 |
| diverse | **0.4200** | [+0.2778, +0.5312] | 100 | 42 | 0 | 2.75 |

**AUC(coherent) − AUC(diverse) = +0.1000**  95% CI (family) [-0.1389, +0.3600]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.
— **AUC(diverse) = 0.4200 is below 0.5**: confidence runs backwards against truth in that condition. That is the crossover, not merely a smaller effect.
— salience gap over the same items: -0.05 of 5 — effectively equalized, so this AUC gap is not a salience artifact in either direction.

## 4. Salience, reported as a covariate

| cell | n with salience | mean | max |
|---|---|---|---|
| `coherent_false` | 20 | 2.83 | 3.00 |
| `diverse_false` | 20 | 2.65 | 3.00 |

_Logistic model not fitted: catch outcome is constant (0/40 caught); a logistic fit is undefined_


## 5. Diagnostics (not endpoints)

> Cell means, the 2x2 effects and the ANOVA are diagnostics. They describe confidence level; the endpoint is the confidence-accuracy relationship, which is the AUC.

### Mean confidence per cell — P(yes) three-way

| cell | n | mean | 95% CI (item) |
|---|---|---|---|
| `coherent_true` | 20 | 0.6888 | [0.6756, 0.7008] |
| `coherent_false` | 20 | 0.6893 | [0.6755, 0.7018] |
| `diverse_true` | 20 | 0.6965 | [0.6802, 0.7121] |
| `diverse_false` | 20 | 0.6983 | [0.6825, 0.7133] |

### Mean confidence per cell — P(yes) two-way (yes vs no only)

| cell | mean | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.6923 | [0.6792, 0.7042] |
| `coherent_false` | 0.6927 | [0.6791, 0.7051] |
| `diverse_true` | 0.6997 | [0.6834, 0.7153] |
| `diverse_false` | 0.7015 | [0.6856, 0.7166] |

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
| main_effect_coherence | -0.0084 | [-0.0229, +0.0065] | [-0.0144, -0.0026] |
| main_effect_truth | -0.0011 | [-0.0156, +0.0131] | [-0.0030, +0.0007] |
| interaction | +0.0013 | [-0.0274, +0.0299] | [-0.0058, +0.0083] |
| coherence_effect_within_true | -0.0078 | [-0.0282, +0.0133] | [-0.0142, -0.0013] |
| coherence_effect_within_false | -0.0091 | [-0.0294, +0.0114] | [-0.0167, -0.0018] |

- **main_effect_coherence** — mean(coherent) - mean(diverse), collapsing over truth
- **main_effect_truth** — mean(true) - mean(false), collapsing over coherence
- **interaction** — (coh_true - coh_false) - (div_true - div_false)
- **coherence_effect_within_true** — D-004: the CLEAN coherence contrast. Both cells are flawless, so neither the flaw-type confound nor flaw salience can touch it.
- **coherence_effect_within_false** — D-004: mixes coherence with whatever differs between the mechanisms.

### Two-way ANOVA-style breakdown

| source | SS | df | MS | F | p | partial eta^2 |
|---|---|---|---|---|---|---|
| coherence | 0.00141 | 1 | 0.00141 | 1.221 | 0.27268 | 0.0158 |
| truth | 0.00003 | 1 | 0.00003 | 0.022 | 0.88359 | 0.0003 |
| coherence x truth | 0.00001 | 1 | 0.00001 | 0.007 | 0.93152 | 0.0001 |
| residual | 0.08806 | 76 | 0.00116 |  |  |  |

> F-tests assume independent observations. Items come in families of 4 sharing a scenario, so they are not independent; treat these as a variance breakdown and read the bootstrap CIs on the effect estimates for inference.

### Coverage

- option mass covered: mean 0.6915, min 0.5957, max 0.7327
- low coverage means the three options hold little of the next-token distribution and the renormalized numbers are ratios of small numbers.
