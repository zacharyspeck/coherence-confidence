# coherence-confidence — analysis

- run: `Qwen_Qwen3-8B__c043a4daa403__46945370bffd59fa`
- model: `Qwen/Qwen3-8B` (revision `b968826d9c46dd6066d109eabc6255188de91218`)
- template hash: `46945370bffd59fa`
- options: `['Yes', 'No', 'Unknown']`
- scored: 100 items in 20 families
- timestamp: 2026-09-06T17:41:19+00:00

## 1. PRIMARY ENDPOINT — AUC(coherent) vs AUC(diverse)

> AUC(coherent) - AUC(diverse) on the FULL set. Koriat's consensuality principle predicts NEGATIVE: coherence raises confidence without raising accuracy, so the confidence-accuracy relationship degrades where the evidence agrees with itself. AUC(coherent) below 0.5 is the crossover.

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5275** | [+0.4450, +0.6226] | 400 | 211 | 0 | — |
| diverse | **0.4550** | [+0.3700, +0.5325] | 400 | 182 | 0 | 2.56 |

**AUC(coherent) − AUC(diverse) = +0.0725**  95% CI (family) [-0.0375, +0.2075]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.
— **AUC(diverse) = 0.4550**: point estimate below 0.5 but the interval spans 0.5: confidence gives no separation between true and false in that condition. The licensed claim is no separation, not reversal.

**Surface-complexity control: The decorative control tracks the COHERENT cells (AUC 0.640 against 0.630 coherent and 0.550 diverse), so the effect is about evidential independence and not about how much there is to parse.**
(decorative AUC is 0.010 from coherent and 0.090 from diverse; section 3 has the table.)

## 2. The same endpoint, confound-controlled

> The same gap on the MATCHED-MECHANISM subset, where both conditions are falsified by scope_mismatch, so flaw mechanism cannot differ between them and only coherence does (D-024).

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.6300** | [+0.5408, +0.8025] | 100 | 63 | 0 | — |
| diverse | **0.5500** | [+0.4074, +0.7100] | 100 | 55 | 0 | 2.56 |

**AUC(coherent) − AUC(diverse) = +0.0800**  95% CI (family) [-0.1094, +0.3333]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.

Both conditions here are falsified by the *same* mechanism, so a gap cannot be attributed to one cell's flaws being a different kind of thing from the other's. This is the number to quote when asked whether the coherent-false items are simply easier to catch.

## 3. Surface-complexity control

> AUC for all three coherence levels inside the 10 families that carry the control arm. A decorative item is as busy to read as a diverse one and as evidentially dependent as a coherent one, so which of the two its AUC sits with says whether the effect is about evidential independence or about parse load (D-030).

| level | AUC | 95% CI (family) | pairs | distinct entities | condition values |
|---|---|---|---|---|---|
| coherent | **0.6300** | [+0.5408, +0.8025] | 100 | 13.2 | 4.0 |
| diverse | **0.5500** | [+0.4074, +0.7100] | 100 | 24.7 | 16.0 |
| decorative | **0.6400** | [+0.5510, +0.7969] | 100 | 25.2 | 4.0 |

Read the last two columns together. The decorative row matches **diverse** on distinct entities and **coherent** on condition values - that is the control working. Which AUC it lands nearer is the result.

- **decorative − coherent = +0.0100**  95% CI (family) [-0.1250, +0.1539]
- **decorative − diverse = +0.0900**  95% CI (family) [-0.0816, +0.3200]

**The decorative control tracks the COHERENT cells (AUC 0.640 against 0.630 coherent and 0.550 diverse), so the effect is about evidential independence and not about how much there is to parse.**

## 4. Per-mechanism breakdown

| cell | stated_confound | broken_chronology | scope_mismatch |
|---|---|---|---|
| `coherent_false` | 10 | 0 | 10 |
| `diverse_false` | 0 | 10 | 10 |

### `stated_confound`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.4900** | [+0.3265, +0.6562] | 100 | 49 | 0 | — |

### `broken_chronology`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| diverse | **0.4000** | [+0.2300, +0.5208] | 100 | 40 | 0 | — |

### `scope_mismatch`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.6300** | [+0.5408, +0.8025] | 100 | 63 | 0 | — |
| diverse | **0.5500** | [+0.4074, +0.7100] | 100 | 55 | 0 | 2.56 |

**AUC(coherent) − AUC(diverse) = +0.0800**  95% CI (family) [-0.1094, +0.3333]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.

## 5. Salience, reported as a covariate

| cell | n with salience | mean | max |
|---|---|---|---|
| `coherent_false` | 0 | — | — |
| `diverse_false` | 9 | 2.56 | 3.00 |

Logistic models of **catch-rate** (caught = p_yes_3way < 0.5 on a FALSE item) over 40 FALSE items in the core cells; overall catch rate 25.0%.

| model | coherent | salience (z) | surface (z) |
|---|---|---|---|
| `coherence_only` | **+1.789** | — | — |
| `plus_reader_catch_rate` | **+1.971** | — | — |
| `plus_reader_catch_rate_and_surface` | **+0.123** | — | -1.104 |

Coherence coefficient across the three: [1.7892, 1.9707, 0.1227]. **It does not hold up under conditioning - read the path above to see which covariate absorbs it.**

> Read the coherence coefficient down the three models. If it keeps its sign and size as the reader catch rate and then surface complexity are added, neither covariate explains the coherence effect. If it collapses toward zero when a covariate enters, that covariate was doing the work.

## 6. Diagnostics (not endpoints)

> Cell means, the 2x2 effects and the ANOVA are diagnostics. They describe confidence level; the endpoint is the confidence-accuracy relationship, which is the AUC.

### Mean confidence per cell — P(yes) three-way

| cell | n | mean | 95% CI (item) |
|---|---|---|---|
| `coherent_true` | 20 | 0.6632 | [0.4673, 0.8391] |
| `coherent_false` | 20 | 0.5847 | [0.3950, 0.7662] |
| `diverse_true` | 20 | 0.8228 | [0.6845, 0.9397] |
| `diverse_false` | 20 | 0.9056 | [0.8156, 0.9765] |
| `decorative_true` | 10 | 0.7440 | [0.4993, 0.9649] |
| `decorative_false` | 10 | 0.6017 | [0.3755, 0.8133] |

### Mean confidence per cell — P(yes) two-way (yes vs no only)

| cell | mean | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.9650 | [0.9296, 0.9921] |
| `coherent_false` | 0.8921 | [0.7749, 0.9875] |
| `diverse_true` | 0.9963 | [0.9919, 0.9994] |
| `diverse_false` | 0.9998 | [0.9996, 0.9999] |
| `decorative_true` | 0.9970 | [0.9919, 0.9998] |
| `decorative_false` | 0.9835 | [0.9586, 0.9991] |

### Abstention rate per cell

Fraction of items where the third option has the highest probability.

| cell | rate | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.3000 | [0.1000, 0.5000] |
| `coherent_false` | 0.4000 | [0.2000, 0.6000] |
| `diverse_true` | 0.1500 | [0.0000, 0.3000] |
| `diverse_false` | 0.1000 | [0.0000, 0.2500] |
| `decorative_true` | 0.3000 | [0.0000, 0.6000] |
| `decorative_false` | 0.4000 | [0.1000, 0.7000] |

- full set, `coherent`: 14 abstained, all INCLUDED in the AUC above (D-003).
- full set, `diverse`: 5 abstained, all INCLUDED in the AUC above (D-003).
- matched subset, `coherent`: 6 abstained, all INCLUDED in the AUC above (D-003).
- matched subset, `diverse`: 1 abstained, all INCLUDED in the AUC above (D-003).

### 2x2 effects on mean confidence

> Read the family-clustered CI for these. The 2x2 is within-family, so clustering keeps the pairing; the item-level interval discards it. See DECISIONS.md D-010.

| effect | estimate | 95% CI (item) | **95% CI (family)** |
|---|---|---|---|
| main_effect_coherence | -0.1765 | [-0.3313, -0.0304] | [-0.3060, -0.0650] |
| main_effect_truth | +0.0267 | [-0.1141, +0.1612] | [-0.0428, +0.0919] |
| interaction | +0.1613 | [-0.1384, +0.4641] | [+0.0413, +0.3146] |
| coherence_effect_within_true | -0.1596 | [-0.3868, +0.0601] | [-0.2868, -0.0502] |
| coherence_effect_within_false | -0.3209 | [-0.5280, -0.1234] | [-0.4808, -0.1766] |

- **main_effect_coherence** — mean(coherent) - mean(diverse), collapsing over truth
- **main_effect_truth** — mean(true) - mean(false), collapsing over coherence
- **interaction** — (coh_true - coh_false) - (div_true - div_false)
- **coherence_effect_within_true** — D-004: the CLEAN coherence contrast. Both cells are flawless, so neither the flaw-type confound nor flaw salience can touch it.
- **coherence_effect_within_false** — D-004: mixes coherence with whatever differs between the mechanisms.

### Two-way ANOVA-style breakdown

| source | SS | df | MS | F | p | partial eta^2 |
|---|---|---|---|---|---|---|
| coherence | 1.15467 | 1 | 1.15467 | 9.224 | 0.00327 | 0.1082 |
| truth | 0.00009 | 1 | 0.00009 | 0.001 | 0.97821 | 0.0000 |
| coherence x truth | 0.13007 | 1 | 0.13007 | 1.039 | 0.31128 | 0.0135 |
| residual | 9.51376 | 76 | 0.12518 |  |  |  |

> F-tests assume independent observations. Items come in families of 4 sharing a scenario, so they are not independent; treat these as a variance breakdown and read the bootstrap CIs on the effect estimates for inference.

### Coverage

- option mass covered: mean 1.0000, min 0.9999, max 1.0000
- low coverage means the three options hold little of the next-token distribution and the renormalized numbers are ratios of small numbers.
