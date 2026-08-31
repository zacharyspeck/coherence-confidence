# Salience convergence log

Mean blind-audit explicitness (1-5) of the flaw in each FALSE cell,
one row per round. Targets: gap within 0.4, no cell above 3.5, and every FALSE item
still 100% findable. Hard stop at 4 rounds - past that, report the gap
rather than editing items until the number lands.

| round | coherent_false | diverse_false | gap | worst cell | gap ok | cell ok | all findable | false positives |
|---|---|---|---|---|---|---|---|---|
| R0 - BEFORE the fix pass (confound in the final sentence) | 3.90 | 2.70 | +1.20 | 3.90 | NO | NO | YES | 0/40 |

<sub>by mechanism: broken_chronology 2.55 (n=10), claim_mismatch 2.85 (n=10), shared_confound 3.90 (n=20). 16 of 40 FALSE items were flagged too-easy; all 16 were coherent_false.</sub>

| R1 - post-rebuild (buried confound + scope_mismatch) | 3.17 | 2.85 | +0.32 | 3.17 | YES | YES | YES | 4/40 |

<sub>by mechanism: broken_chronology 2.75 (n=10), scope_mismatch 2.98 (n=20), stated_confound 3.35 (n=10)</sub>

