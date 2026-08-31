# coherence-confidence — analysis

- run: `HuggingFaceTB_SmolLM2-135M__107ab946de39__cd91ee96dc31dce3`
- model: `HuggingFaceTB/SmolLM2-135M` (revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`)
- template hash: `cd91ee96dc31dce3`
- options: `['Yes', 'No', 'Unknown']`
- scored: 80 items in 20 families
- timestamp: 2026-08-31T08:02:22+00:00

## Policy

- **abstained_items_in_auc**: INCLUDED (D-003) using p_yes_3way
- **auc_scope**: WITHIN each coherence condition, never pooled
- **auc_implementation**: explicit pairwise win rate, ties = 0.5
- **primary_measure**: p_yes_3way = p_yes / (p_yes + p_no + p_unsure)
- **bootstrap**: 10000 resamples, percentile 95% CI, two units: stratified-by-item and clustered-by-family (D-010)
- **which_ci_to_read**: Per-cell numbers (cell means, abstention rates, within-condition AUC): ci_item and ci_family are IDENTICAL by construction, since each family contributes exactly one item per cell. For the cross-cell contrasts (main_effect_coherence, interaction, coherence_effect_within_*): read ci_family. The 2x2 is within-family, so the clustered resample keeps the pairing and ci_item throws that pairing away.

## Cell means — P(yes) three-way

| cell | n | mean | 95% CI (item) | 95% CI (family) |
|---|---|---|---|---|
| `coherent_true` | 20 | 0.6385 | [0.6138, 0.6635] | [0.6141, 0.6632] |
| `coherent_false` | 20 | 0.6386 | [0.6147, 0.6629] | [0.6151, 0.6630] |
| `diverse_true` | 20 | 0.6418 | [0.6181, 0.6662] | [0.6184, 0.6670] |
| `diverse_false` | 20 | 0.6432 | [0.6191, 0.6671] | [0.6197, 0.6675] |

## Cell means — P(yes) two-way (yes vs no only)

| cell | mean | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.6397 | [0.6150, 0.6646] |
| `coherent_false` | 0.6397 | [0.6158, 0.6640] |
| `diverse_true` | 0.6428 | [0.6191, 0.6672] |
| `diverse_false` | 0.6442 | [0.6201, 0.6681] |

## AUC within condition

20 true vs 20 false per condition = 400 pairs. Never pooled.

| condition | AUC | 95% CI (item) | 95% CI (family) | pairs | wins | ties |
|---|---|---|---|---|---|---|
| coherent | 0.5125 | [0.3300, 0.6975] | [0.4500, 0.5825] | 400 | 205 | 0 |
| diverse | 0.4750 | [0.2950, 0.6525] | [0.4125, 0.5300] | 400 | 190 | 0 |


## Abstention rate per cell

Fraction of items where the third option has the highest probability.

| cell | rate | 95% CI (item) |
|---|---|---|
| `coherent_true` | 0.0000 | [0.0000, 0.0000] |
| `coherent_false` | 0.0000 | [0.0000, 0.0000] |
| `diverse_true` | 0.0000 | [0.0000, 0.0000] |
| `diverse_false` | 0.0000 | [0.0000, 0.0000] |

## 2x2 effects

> **Read the family-clustered CI for these.** The 2x2 is within-family, so clustering keeps the pairing; the item-level interval discards it and comes out too wide. (For the per-cell tables above, the two CIs are identical by construction.) See DECISIONS.md D-010.

| effect | estimate | 95% CI (item) | **95% CI (family)** |
|---|---|---|---|
| main_effect_coherence | -0.0039 | [-0.0288, +0.0203] | [-0.0129, +0.0046] |
| main_effect_truth | -0.0007 | [-0.0250, +0.0237] | [-0.0025, +0.0010] |
| interaction | +0.0012 | [-0.0463, +0.0496] | [-0.0088, +0.0113] |
| coherence_effect_within_true | -0.0033 | [-0.0379, +0.0320] | [-0.0133, +0.0062] |
| coherence_effect_within_false | -0.0045 | [-0.0382, +0.0297] | [-0.0149, +0.0054] |

- **main_effect_coherence** — mean(coherent) - mean(diverse), collapsing over truth
- **main_effect_truth** — mean(true) - mean(false), collapsing over coherence
- **interaction** — (coh_true - coh_false) - (div_true - div_false)
- **coherence_effect_within_true** — D-004: the CLEAN coherence contrast. Both cells are flawless, so the flaw-type confound between coherent_false and diverse_false cannot touch this one.
- **coherence_effect_within_false** — D-004: CONFOUNDED. coherent_false is broken by a shared confound and diverse_false by bad dates or claim mismatch, so this mixes coherence with flaw type.

## Two-way ANOVA-style breakdown

| source | SS | df | MS | F | p | partial eta^2 |
|---|---|---|---|---|---|---|
| coherence | 0.00030 | 1 | 0.00030 | 0.094 | 0.76060 | 0.0012 |
| truth | 0.00001 | 1 | 0.00001 | 0.003 | 0.95497 | 0.0000 |
| coherence x truth | 0.00001 | 1 | 0.00001 | 0.002 | 0.96163 | 0.0000 |
| residual | 0.24691 | 76 | 0.00325 |  |  |  |

> F-tests assume independent observations. Items come in families of 4 sharing a scenario, so they are not independent; treat these as a variance breakdown and read the bootstrap CIs on the effect estimates for inference.

## diverse_false, split by flaw type (D-004)

| flaw_type | n | mean P(yes) |
|---|---|---|
| temporal | 10 | 0.6148 |
| claim_mismatch | 10 | 0.6715 |

## Coverage

- option mass covered: mean 0.7326, min 0.6399, max 0.7955
- low coverage means the three options hold little of the next-token distribution and the renormalized numbers are ratios of small numbers.
