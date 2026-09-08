# coherence-confidence — analysis

- run: `unsloth_Qwen3-32B-bnb-4bit__c043a4daa403__46945370bffd59fa`
- model: `unsloth/Qwen3-32B-bnb-4bit` (revision `7f721e74a6a8cc9ee352f7e49303a2c1705f9083`)
- template hash: `46945370bffd59fa`
- options: `['Yes', 'No', 'Unknown']`
- scored: 100 items in 20 families
- timestamp: 2026-09-06T17:55:25+00:00

## 1. PRIMARY ENDPOINT — AUC(coherent) vs AUC(diverse)

> AUC(coherent) - AUC(diverse) on the FULL set. Koriat's consensuality principle predicts NEGATIVE: coherence raises confidence without raising accuracy, so the confidence-accuracy relationship degrades where the evidence agrees with itself. AUC(coherent) below 0.5 is the crossover.

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.6175** | [+0.5000, +0.7375] | 400 | 247 | 0 | — |
| diverse | **0.3925** | [+0.2700, +0.5150] | 400 | 157 | 0 | 2.56 |

**AUC(coherent) − AUC(diverse) = +0.2250**  95% CI (family) [+0.0250, +0.4350]
— sign is OPPOSITE to the consensuality prediction; the interval excludes zero.
— **AUC(diverse) = 0.3925**: point estimate below 0.5 but the interval spans 0.5: confidence gives no separation between true and false in that condition. The licensed claim is no separation, not reversal.

**Surface-complexity control: The decorative control tracks the COHERENT cells (AUC 0.660 against 0.760 coherent and 0.410 diverse), so the effect is about evidential independence and not about how much there is to parse.**
(decorative AUC is 0.100 from coherent and 0.250 from diverse; section 3 has the table.)

## 2. The same endpoint, confound-controlled

> The same gap on the MATCHED-MECHANISM subset, where both conditions are falsified by scope_mismatch, so flaw mechanism cannot differ between them and only coherence does (D-024).

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.7600** | [+0.6694, +0.9388] | 100 | 76 | 0 | — |
| diverse | **0.4100** | [+0.2222, +0.6122] | 100 | 41 | 0 | 2.56 |

**AUC(coherent) − AUC(diverse) = +0.3500**  95% CI (family) [+0.0988, +0.6735]
— sign is OPPOSITE to the consensuality prediction; the interval excludes zero.
— **AUC(diverse) = 0.4100**: point estimate below 0.5 but the interval spans 0.5: confidence gives no separation between true and false in that condition. The licensed claim is no separation, not reversal.

Both conditions here are falsified by the *same* mechanism, so a gap cannot be attributed to one cell's flaws being a different kind of thing from the other's. This is the number to quote when asked whether the coherent-false items are simply easier to catch.

## 3. Surface-complexity control

> AUC for all three coherence levels inside the 10 families that carry the control arm. A decorative item is as busy to read as a diverse one and as evidentially dependent as a coherent one, so which of the two its AUC sits with says whether the effect is about evidential independence or about parse load (D-030).

| level | AUC | 95% CI (family) | pairs | distinct entities | condition values |
|---|---|---|---|---|---|
| coherent | **0.7600** | [+0.6694, +0.9388] | 100 | 13.2 | 4.0 |
| diverse | **0.4100** | [+0.2222, +0.6122] | 100 | 24.7 | 16.0 |
| decorative | **0.6600** | [+0.5200, +0.8519] | 100 | 25.2 | 4.0 |

Read the last two columns together. The decorative row matches **diverse** on distinct entities and **coherent** on condition values - that is the control working. Which AUC it lands nearer is the result.

- **decorative − coherent = -0.1000**  95% CI (family) [-0.3306, +0.0781]
- **decorative − diverse = +0.2500**  95% CI (family) [+0.0000, +0.5313]

**The decorative control tracks the COHERENT cells (AUC 0.660 against 0.760 coherent and 0.410 diverse), so the effect is about evidential independence and not about how much there is to parse.**

## 4. Per-mechanism breakdown

| cell | stated_confound | broken_chronology | scope_mismatch |
|---|---|---|---|
| `coherent_false` | 10 | 0 | 10 |
| `diverse_false` | 0 | 10 | 10 |

### `stated_confound`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.4400** | [+0.2344, +0.6033] | 100 | 44 | 0 | — |

### `broken_chronology`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| diverse | **0.3900** | [+0.1837, +0.5918] | 100 | 39 | 0 | — |

### `scope_mismatch`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.7600** | [+0.6694, +0.9388] | 100 | 76 | 0 | — |
| diverse | **0.4100** | [+0.2222, +0.6122] | 100 | 41 | 0 | 2.56 |

**AUC(coherent) − AUC(diverse) = +0.3500**  95% CI (family) [+0.0988, +0.6735]
— sign is OPPOSITE to the consensuality prediction; the interval excludes zero.
— **AUC(diverse) = 0.4100**: point estimate below 0.5 but the interval spans 0.5: confidence gives no separation between true and false in that condition. The licensed claim is no separation, not reversal.

## 5. Salience, reported as a covariate

| cell | n with salience | mean | max |
|---|---|---|---|
| `coherent_false` | 0 | — | — |
| `diverse_false` | 9 | 2.56 | 3.00 |

Logistic models of **catch-rate** (caught = p_yes_3way < 0.5 on a FALSE item) over 40 FALSE items in the core cells; overall catch rate 42.5%.

| model | coherent | salience (z) | surface (z) |
|---|---|---|---|
| `coherence_only` | **+1.504** | — | — |
| `plus_reader_catch_rate` | **+1.556** | — | — |
| `plus_reader_catch_rate_and_surface` | **+2.310** | — | +0.422 |

Coherence coefficient across the three: [1.5041, 1.5556, 2.3098]. **It holds its sign and size, so neither salience nor surface complexity explains it.**

> Read the coherence coefficient down the three models. If it keeps its sign and size as the reader catch rate and then surface complexity are added, neither covariate explains the coherence effect. If it collapses toward zero when a covariate enters, that covariate was doing the work.

## 6. Diagnostics (not endpoints)

> Cell means, the 2x2 effects and the ANOVA are diagnostics. They describe confidence level; the endpoint is the confidence-accuracy relationship, which is the AUC.

### Mean confidence per cell — P(yes) three-way

| cell | n | mean | 95% CI (item) |
|---|---|---|---|
| `coherent_true` | 20 | 0.5262 | [0.3518, 0.6948] |
| `coherent_false` | 20 | 0.3799 | [0.2433, 0.5208] |
| `diverse_true` | 20 | 0.6334 | [0.5010, 0.7592] |
| `diverse_false` | 20 | 0.7302 | [0.5964, 0.8489] |
| `decorative_true` | 10 | 0.6873 | [0.4675, 0.8766] |
| `decorative_false` | 10 | 0.5557 | [0.3570, 0.7390] |

### Mean confidence per cell — P(yes) two-way (yes vs no only)

| cell | mean | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.8129 | [0.6922, 0.9161] |
| `coherent_false` | 0.7397 | [0.6018, 0.8649] |
| `diverse_true` | 0.9324 | [0.8582, 0.9764] |
| `diverse_false` | 0.9591 | [0.9286, 0.9844] |
| `decorative_true` | 0.9262 | [0.8209, 0.9925] |
| `decorative_false` | 0.8845 | [0.7510, 0.9820] |

### Abstention rate per cell

Fraction of items where the third option has the highest probability.

| cell | rate | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.4500 | [0.2500, 0.6500] |
| `coherent_false` | 0.6000 | [0.4000, 0.8000] |
| `diverse_true` | 0.3500 | [0.1500, 0.5500] |
| `diverse_false` | 0.2500 | [0.0500, 0.4500] |
| `decorative_true` | 0.3000 | [0.0000, 0.6000] |
| `decorative_false` | 0.3000 | [0.0000, 0.6000] |

- full set, `coherent`: 21 abstained, all INCLUDED in the AUC above (D-003).
- full set, `diverse`: 12 abstained, all INCLUDED in the AUC above (D-003).
- matched subset, `coherent`: 8 abstained, all INCLUDED in the AUC above (D-003).
- matched subset, `diverse`: 5 abstained, all INCLUDED in the AUC above (D-003).

### 2x2 effects on mean confidence

> Read the family-clustered CI for these. The 2x2 is within-family, so clustering keeps the pairing; the item-level interval discards it. See DECISIONS.md D-010.

| effect | estimate | 95% CI (item) | **95% CI (family)** |
|---|---|---|---|
| main_effect_coherence | -0.2086 | [-0.3434, -0.0744] | [-0.3197, -0.1043] |
| main_effect_truth | +0.0461 | [-0.0822, +0.1719] | [-0.0096, +0.0960] |
| interaction | +0.2430 | [-0.0444, +0.5235] | [+0.0482, +0.4543] |
| coherence_effect_within_true | -0.1072 | [-0.3229, +0.1046] | [-0.2659, +0.0403] |
| coherence_effect_within_false | -0.3503 | [-0.5348, -0.1570] | [-0.4775, -0.2293] |

- **main_effect_coherence** — mean(coherent) - mean(diverse), collapsing over truth
- **main_effect_truth** — mean(true) - mean(false), collapsing over coherence
- **interaction** — (coh_true - coh_false) - (div_true - div_false)
- **coherence_effect_within_true** — D-004: the CLEAN coherence contrast. Both cells are flawless, so neither the flaw-type confound nor flaw salience can touch it.
- **coherence_effect_within_false** — D-004: mixes coherence with whatever differs between the mechanisms.

### Two-way ANOVA-style breakdown

| source | SS | df | MS | F | p | partial eta^2 |
|---|---|---|---|---|---|---|
| coherence | 1.04649 | 1 | 1.04649 | 9.285 | 0.00318 | 0.1089 |
| truth | 0.01225 | 1 | 0.01225 | 0.109 | 0.74255 | 0.0014 |
| coherence x truth | 0.29535 | 1 | 0.29535 | 2.620 | 0.10964 | 0.0333 |
| residual | 8.56599 | 76 | 0.11271 |  |  |  |

> F-tests assume independent observations. Items come in families of 4 sharing a scenario, so they are not independent; treat these as a variance breakdown and read the bootstrap CIs on the effect estimates for inference.

### Coverage

- option mass covered: mean 0.9632, min 0.9226, max 0.9907
- low coverage means the three options hold little of the next-token distribution and the renormalized numbers are ratios of small numbers.
