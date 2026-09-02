# coherence-confidence — analysis

- run: `Qwen_Qwen2.5-3B-Instruct__c043a4daa403__2f392d0ec884fc6a`
- model: `Qwen/Qwen2.5-3B-Instruct` (revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`)
- template hash: `2f392d0ec884fc6a`
- options: `['Yes', 'No', 'Unsure']`
- scored: 100 items in 20 families
- timestamp: 2026-09-02T13:47:54+00:00

## 1. PRIMARY ENDPOINT — AUC(coherent) vs AUC(diverse)

> AUC(coherent) - AUC(diverse) on the FULL set. Koriat's consensuality principle predicts NEGATIVE: coherence raises confidence without raising accuracy, so the confidence-accuracy relationship degrades where the evidence agrees with itself. AUC(coherent) below 0.5 is the crossover.

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5550** | [+0.4475, +0.6675] | 400 | 222 | 0 | — |
| diverse | **0.4475** | [+0.3525, +0.5400] | 400 | 179 | 0 | 2.56 |

**AUC(coherent) − AUC(diverse) = +0.1075**  95% CI (family) [-0.0375, +0.2475]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.
— **AUC(diverse) = 0.4475 is below 0.5**: confidence runs backwards against truth in that condition. That is the crossover, not merely a smaller effect.

**Surface-complexity control: The decorative control tracks the DIVERSE cells (AUC 0.580 against 0.520 diverse and 0.690 coherent), so the effect is about surface complexity and the evidential-independence story is wrong.**
(decorative AUC is 0.110 from coherent and 0.060 from diverse; section 3 has the table.)

## 2. The same endpoint, confound-controlled

> The same gap on the MATCHED-MECHANISM subset, where both conditions are falsified by scope_mismatch, so flaw mechanism cannot differ between them and only coherence does (D-024).

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.6900** | [+0.5000, +0.8889] | 100 | 69 | 0 | — |
| diverse | **0.5200** | [+0.3438, +0.7200] | 100 | 52 | 0 | 2.56 |

**AUC(coherent) − AUC(diverse) = +0.1700**  95% CI (family) [-0.1562, +0.4531]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.

Both conditions here are falsified by the *same* mechanism, so a gap cannot be attributed to one cell's flaws being a different kind of thing from the other's. This is the number to quote when asked whether the coherent-false items are simply easier to catch.

## 3. Surface-complexity control

> AUC for all three coherence levels inside the 10 families that carry the control arm. A decorative item is as busy to read as a diverse one and as evidentially dependent as a coherent one, so which of the two its AUC sits with says whether the effect is about evidential independence or about parse load (D-030).

| level | AUC | 95% CI (family) | pairs | distinct entities | condition values |
|---|---|---|---|---|---|
| coherent | **0.6900** | [+0.5000, +0.8889] | 100 | 13.2 | 4.0 |
| diverse | **0.5200** | [+0.3438, +0.7200] | 100 | 24.7 | 16.0 |
| decorative | **0.5800** | [+0.4688, +0.7408] | 100 | 25.2 | 4.0 |

Read the last two columns together. The decorative row matches **diverse** on distinct entities and **coherent** on condition values - that is the control working. Which AUC it lands nearer is the result.

- **decorative − coherent = -0.1100**  95% CI (family) [-0.3200, +0.1251]
- **decorative − diverse = +0.0600**  95% CI (family) [-0.1406, +0.3600]

**The decorative control tracks the DIVERSE cells (AUC 0.580 against 0.520 diverse and 0.690 coherent), so the effect is about surface complexity and the evidential-independence story is wrong.**

## 4. Per-mechanism breakdown

| cell | stated_confound | broken_chronology | scope_mismatch |
|---|---|---|---|
| `coherent_false` | 10 | 0 | 10 |
| `diverse_false` | 0 | 10 | 10 |

### `stated_confound`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.4800** | [+0.2900, +0.6800] | 100 | 48 | 0 | — |

### `broken_chronology`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| diverse | **0.4300** | [+0.2400, +0.5918] | 100 | 43 | 0 | — |

### `scope_mismatch`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.6900** | [+0.5000, +0.8889] | 100 | 69 | 0 | — |
| diverse | **0.5200** | [+0.3438, +0.7200] | 100 | 52 | 0 | 2.56 |

**AUC(coherent) − AUC(diverse) = +0.1700**  95% CI (family) [-0.1562, +0.4531]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.

## 5. Salience, reported as a covariate

| cell | n with salience | mean | max |
|---|---|---|---|
| `coherent_false` | 0 | — | — |
| `diverse_false` | 9 | 2.56 | 3.00 |

Logistic models of **catch-rate** (caught = p_yes_3way < 0.5 on a FALSE item) over 40 FALSE items in the core cells; overall catch rate 62.5%.

| model | coherent | salience (z) | surface (z) |
|---|---|---|---|
| `coherence_only` | **+0.647** | — | — |
| `plus_reader_catch_rate` | **+0.690** | — | — |
| `plus_reader_catch_rate_and_surface` | **-0.266** | — | -0.558 |

Coherence coefficient across the three: [0.6467, 0.69, -0.2659]. **It does not hold up under conditioning - read the path above to see which covariate absorbs it.**

> Read the coherence coefficient down the three models. If it keeps its sign and size as the reader catch rate and then surface complexity are added, neither covariate explains the coherence effect. If it collapses toward zero when a covariate enters, that covariate was doing the work.

## 6. Diagnostics (not endpoints)

> Cell means, the 2x2 effects and the ANOVA are diagnostics. They describe confidence level; the endpoint is the confidence-accuracy relationship, which is the AUC.

### Mean confidence per cell — P(yes) three-way

| cell | n | mean | 95% CI (item) |
|---|---|---|---|
| `coherent_true` | 20 | 0.4977 | [0.3513, 0.6392] |
| `coherent_false` | 20 | 0.4392 | [0.3160, 0.5659] |
| `diverse_true` | 20 | 0.3461 | [0.2219, 0.4820] |
| `diverse_false` | 20 | 0.4169 | [0.2683, 0.5758] |
| `decorative_true` | 10 | 0.7459 | [0.5670, 0.8847] |
| `decorative_false` | 10 | 0.6350 | [0.4083, 0.8425] |

### Mean confidence per cell — P(yes) two-way (yes vs no only)

| cell | mean | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.4977 | [0.3513, 0.6392] |
| `coherent_false` | 0.4392 | [0.3160, 0.5659] |
| `diverse_true` | 0.3461 | [0.2219, 0.4821] |
| `diverse_false` | 0.4170 | [0.2683, 0.5758] |
| `decorative_true` | 0.7459 | [0.5670, 0.8847] |
| `decorative_false` | 0.6350 | [0.4083, 0.8425] |

### Abstention rate per cell

Fraction of items where the third option has the highest probability.

| cell | rate | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.0000 | [0.0000, 0.0000] |
| `coherent_false` | 0.0000 | [0.0000, 0.0000] |
| `diverse_true` | 0.0000 | [0.0000, 0.0000] |
| `diverse_false` | 0.0000 | [0.0000, 0.0000] |
| `decorative_true` | 0.0000 | [0.0000, 0.0000] |
| `decorative_false` | 0.0000 | [0.0000, 0.0000] |


### 2x2 effects on mean confidence

> Read the family-clustered CI for these. The 2x2 is within-family, so clustering keeps the pairing; the item-level interval discards it. See DECISIONS.md D-010.

| effect | estimate | 95% CI (item) | **95% CI (family)** |
|---|---|---|---|
| main_effect_coherence | -0.0161 | [-0.1436, +0.1078] | [-0.1475, +0.1223] |
| main_effect_truth | +0.0172 | [-0.1094, +0.1378] | [-0.0588, +0.0867] |
| interaction | +0.1293 | [-0.1512, +0.4021] | [-0.0149, +0.2631] |
| coherence_effect_within_true | +0.1516 | [-0.0507, +0.3404] | [-0.0482, +0.3426] |
| coherence_effect_within_false | +0.0223 | [-0.1783, +0.2224] | [-0.1427, +0.1922] |

- **main_effect_coherence** — mean(coherent) - mean(diverse), collapsing over truth
- **main_effect_truth** — mean(true) - mean(false), collapsing over coherence
- **interaction** — (coh_true - coh_false) - (div_true - div_false)
- **coherence_effect_within_true** — D-004: the CLEAN coherence contrast. Both cells are flawless, so neither the flaw-type confound nor flaw salience can touch it.
- **coherence_effect_within_false** — D-004: mixes coherence with whatever differs between the mechanisms.

### Two-way ANOVA-style breakdown

| source | SS | df | MS | F | p | partial eta^2 |
|---|---|---|---|---|---|---|
| coherence | 0.15109 | 1 | 0.15109 | 1.433 | 0.23505 | 0.0185 |
| truth | 0.00076 | 1 | 0.00076 | 0.007 | 0.93237 | 0.0001 |
| coherence x truth | 0.08360 | 1 | 0.08360 | 0.793 | 0.37608 | 0.0103 |
| residual | 8.01509 | 76 | 0.10546 |  |  |  |

> F-tests assume independent observations. Items come in families of 4 sharing a scenario, so they are not independent; treat these as a variance breakdown and read the bootstrap CIs on the effect estimates for inference.

### Coverage

- option mass covered: mean 0.4858, min 0.0369, max 0.9997
- low coverage means the three options hold little of the next-token distribution and the renormalized numbers are ratios of small numbers.
