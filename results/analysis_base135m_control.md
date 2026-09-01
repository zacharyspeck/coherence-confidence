# coherence-confidence — analysis

- run: `HuggingFaceTB_SmolLM2-135M__c043a4daa403__efdbe3c2fc149d59`
- model: `HuggingFaceTB/SmolLM2-135M` (revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`)
- template hash: `efdbe3c2fc149d59`
- options: `['Yes', 'No', 'Maybe']`
- scored: 100 items in 20 families
- timestamp: 2026-09-01T04:06:55+00:00

## 1. PRIMARY ENDPOINT — AUC(coherent) vs AUC(diverse)

> AUC(coherent) - AUC(diverse) on the FULL set. Koriat's consensuality principle predicts NEGATIVE: coherence raises confidence without raising accuracy, so the confidence-accuracy relationship degrades where the evidence agrees with itself. AUC(coherent) below 0.5 is the crossover.

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5275** | [+0.4500, +0.6126] | 400 | 211 | 0 | 2.83 |
| diverse | **0.4875** | [+0.4250, +0.5500] | 400 | 195 | 0 | 2.55 |

**AUC(coherent) − AUC(diverse) = +0.0400**  95% CI (family) [-0.0750, +0.1675]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.
— **AUC(diverse) = 0.4875 is below 0.5**: confidence runs backwards against truth in that condition. That is the crossover, not merely a smaller effect.
— salience gap over the same items: +0.28 of 5 — the coherent flaws were louder, which pushes this AUC gap upward and therefore *against* the hypothesis - the test is conservative here.

**Surface-complexity control: The decorative control sits between the coherent and diverse cells (AUC 0.500 against 0.560 coherent and 0.430 diverse), so this run does not separate evidential independence from surface complexity.**
(decorative AUC is 0.060 from coherent and 0.070 from diverse; section 3 has the table.)

## 2. The same endpoint, confound-controlled

> The same gap on the MATCHED-MECHANISM subset, where both conditions are falsified by scope_mismatch, so flaw mechanism cannot differ between them and only coherence does (D-024).

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5600** | [+0.4062, +0.7273] | 100 | 56 | 0 | 2.70 |
| diverse | **0.4300** | [+0.2917, +0.5432] | 100 | 43 | 0 | 2.60 |

**AUC(coherent) − AUC(diverse) = +0.1300**  95% CI (family) [-0.1200, +0.4000]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.
— **AUC(diverse) = 0.4300 is below 0.5**: confidence runs backwards against truth in that condition. That is the crossover, not merely a smaller effect.
— salience gap over the same items: +0.10 of 5 — the coherent flaws were louder, which pushes this AUC gap upward and therefore *against* the hypothesis - the test is conservative here.

Both conditions here are falsified by the *same* mechanism, so a gap cannot be attributed to one cell's flaws being a different kind of thing from the other's. This is the number to quote when asked whether the coherent-false items are simply easier to catch.

## 3. Surface-complexity control

> AUC for all three coherence levels inside the 10 families that carry the control arm. A decorative item is as busy to read as a diverse one and as evidentially dependent as a coherent one, so which of the two its AUC sits with says whether the effect is about evidential independence or about parse load (D-030).

| level | AUC | 95% CI (family) | pairs | distinct entities | condition values |
|---|---|---|---|---|---|
| coherent | **0.5600** | [+0.4062, +0.7273] | 100 | 13.2 | 4.0 |
| diverse | **0.4300** | [+0.2917, +0.5432] | 100 | 24.7 | 16.0 |
| decorative | **0.5000** | [+0.3889, +0.6111] | 100 | 25.2 | 4.0 |

Read the last two columns together. The decorative row matches **diverse** on distinct entities and **coherent** on condition values - that is the control working. Which AUC it lands nearer is the result.

- **decorative − coherent = -0.0600**  95% CI (family) [-0.1975, +0.0781]
- **decorative − diverse = +0.0700**  95% CI (family) [-0.1300, +0.2986]

**The decorative control sits between the coherent and diverse cells (AUC 0.500 against 0.560 coherent and 0.430 diverse), so this run does not separate evidential independence from surface complexity.**

## 4. Per-mechanism breakdown

| cell | stated_confound | broken_chronology | scope_mismatch |
|---|---|---|---|
| `coherent_false` | 10 | 0 | 10 |
| `diverse_false` | 0 | 10 | 10 |

### `stated_confound`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5000** | [+0.3958, +0.6049] | 100 | 50 | 0 | 2.95 |

### `broken_chronology`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| diverse | **0.5300** | [+0.4200, +0.6667] | 100 | 53 | 0 | 2.50 |

### `scope_mismatch`

| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |
|---|---|---|---|---|---|---|
| coherent | **0.5600** | [+0.4062, +0.7273] | 100 | 56 | 0 | 2.70 |
| diverse | **0.4300** | [+0.2917, +0.5432] | 100 | 43 | 0 | 2.60 |

**AUC(coherent) − AUC(diverse) = +0.1300**  95% CI (family) [-0.1200, +0.4000]
— sign is OPPOSITE to the consensuality prediction; the interval spans zero, so the direction is not resolved.
— **AUC(diverse) = 0.4300 is below 0.5**: confidence runs backwards against truth in that condition. That is the crossover, not merely a smaller effect.
— salience gap over the same items: +0.10 of 5 — the coherent flaws were louder, which pushes this AUC gap upward and therefore *against* the hypothesis - the test is conservative here.

## 5. Salience, reported as a covariate

| cell | n with salience | mean | max |
|---|---|---|---|
| `coherent_false` | 20 | 2.83 | 3.00 |
| `diverse_false` | 20 | 2.55 | 3.00 |

_Conditioning models not fitted: catch outcome is constant (0/40 caught); a logistic fit is undefined_


## 6. Diagnostics (not endpoints)

> Cell means, the 2x2 effects and the ANOVA are diagnostics. They describe confidence level; the endpoint is the confidence-accuracy relationship, which is the AUC.

### Mean confidence per cell — P(yes) three-way

| cell | n | mean | 95% CI (item) |
|---|---|---|---|
| `coherent_true` | 20 | 0.6907 | [0.6767, 0.7031] |
| `coherent_false` | 20 | 0.6896 | [0.6758, 0.7022] |
| `diverse_true` | 20 | 0.6967 | [0.6802, 0.7123] |
| `diverse_false` | 20 | 0.6984 | [0.6825, 0.7134] |
| `decorative_true` | 10 | 0.6700 | [0.6573, 0.6851] |
| `decorative_false` | 10 | 0.6698 | [0.6571, 0.6851] |

### Mean confidence per cell — P(yes) two-way (yes vs no only)

| cell | mean | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.6941 | [0.6803, 0.7065] |
| `coherent_false` | 0.6930 | [0.6794, 0.7055] |
| `diverse_true` | 0.6999 | [0.6835, 0.7155] |
| `diverse_false` | 0.7016 | [0.6858, 0.7166] |
| `decorative_true` | 0.6730 | [0.6601, 0.6885] |
| `decorative_false` | 0.6730 | [0.6600, 0.6885] |

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
| main_effect_coherence | +0.0018 | [-0.0108, +0.0141] | [-0.0056, +0.0087] |
| main_effect_truth | -0.0002 | [-0.0125, +0.0120] | [-0.0022, +0.0016] |
| interaction | +0.0028 | [-0.0264, +0.0311] | [-0.0043, +0.0102] |
| coherence_effect_within_true | -0.0060 | [-0.0269, +0.0144] | [-0.0129, +0.0004] |
| coherence_effect_within_false | -0.0089 | [-0.0291, +0.0119] | [-0.0166, -0.0015] |

- **main_effect_coherence** — mean(coherent) - mean(diverse), collapsing over truth
- **main_effect_truth** — mean(true) - mean(false), collapsing over coherence
- **interaction** — (coh_true - coh_false) - (div_true - div_false)
- **coherence_effect_within_true** — D-004: the CLEAN coherence contrast. Both cells are flawless, so neither the flaw-type confound nor flaw salience can touch it.
- **coherence_effect_within_false** — D-004: mixes coherence with whatever differs between the mechanisms.

### Two-way ANOVA-style breakdown

| source | SS | df | MS | F | p | partial eta^2 |
|---|---|---|---|---|---|---|
| coherence | 0.00110 | 1 | 0.00110 | 0.934 | 0.33677 | 0.0121 |
| truth | 0.00000 | 1 | 0.00000 | 0.002 | 0.96506 | 0.0000 |
| coherence x truth | 0.00004 | 1 | 0.00004 | 0.034 | 0.85364 | 0.0005 |
| residual | 0.08977 | 76 | 0.00118 |  |  |  |

> F-tests assume independent observations. Items come in families of 4 sharing a scenario, so they are not independent; treat these as a variance breakdown and read the bootstrap CIs on the effect estimates for inference.

### Coverage

- option mass covered: mean 0.6919, min 0.5957, max 0.7327
- low coverage means the three options hold little of the next-token distribution and the renormalized numbers are ratios of small numbers.
