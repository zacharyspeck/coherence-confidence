# Item audit — blind flaw identification

Step 9. Every item was shown to independent auditors carrying **only** the claim, the passage and the forced question — no cell label, no `ground_truth`, no `confound_note`, and never two items from the same family in one batch. The TRUE items were mixed in as decoys, so an auditor could not know that everything it saw was broken.

## Headline

| | count | of | rate |
|---|---|---|---|
| FALSE items whose intended flaw was found | 40 | 40 | 100% |
| **FALSE items where NO auditor found the flaw — BROKEN** | **0** | 40 | 0% |
| **FALSE items rated as good as stated outright — TOO EASY** | **1** | 40 | 2% |
| TRUE items an auditor invented a flaw for (false positives) | 4 | 40 | 10% |

**How to read the false-positive rate.** Auditors reported a flaw in 4 of 40 items that have nothing wrong with them. Finding a flaw in a FALSE item is only as informative as that rate is low; treat it as the audit's own noise floor, and discount the detection rate above accordingly.

**How to read the 'too easy' count.** The threshold is a mean explicitness of 4.0 out of 5, where 4 means "obvious; the passage all but says it". That bar is arguably unfair to `coherent_false` by construction: the design *requires* the shared confound to be stated in the passage, so an auditor will always see it once they look. Nothing scored 5 ("the passage states the problem outright"). The number to act on is not this count but the asymmetry below.

## The asymmetry that matters

| group | n | mean explicitness (1-5) |
|---|---|---|
| flaw_mechanism `broken_chronology` | 10 | 2.75 |
| flaw_mechanism `scope_mismatch` | 20 | 2.98 |
| flaw_mechanism `stated_confound` | 10 | 3.35 |
| cell `coherent_false` | 20 | **3.17** |
| cell `diverse_false` | 20 | **2.85** |

**Gap: +0.32 points of 5** between the two FALSE cells — inside the 0.4 target.

This is what the fix pass was for. The flaw used to sit in the final sentence of the passage and the gap was **1.20**; it now sits mid-passage with two cases and a closer after it, and `scope_mismatch` — a mechanism that works identically under both coherence conditions — supplies half of each FALSE cell. `results/salience_log.md` has the round-by-round record.

**Why a residual gap is expected and cannot be driven to zero.** `stated_confound` has to be *stated* in the passage to be a shared confound at all, so it will always be somewhat easier to spot than a date that runs backwards or a population that is quietly narrower. That is D-004, and it is a property of the design rather than of the wording.

**What removes the residual entirely.** The matched-mechanism subset: 10 `scope_mismatch` items in each FALSE cell, drawn from the same 10 families, so the mechanism is identical on both sides and only coherence differs. `src/analyze.py` reports the primary endpoint on that subset directly beneath the full-set version, and carries mean salience beside every AUC plus a logistic model of catch-rate on salience.

On the matched subset the mechanism is `scope_mismatch` on both sides, mean salience 2.98, so the salience difference between conditions there is whatever remains after mechanism is held fixed — reported per condition in the analysis output.

## FLAGGED - broken items (auditors could only gesture, not identify)

_None._

## FLAGGED — broken items (no auditor found the intended flaw)

_None._

## FLAGGED — too easy (flaw rated as good as stated outright)

Explicitness >= 4 of 5 means auditors felt the passage announced the problem. These items measure reading rather than reasoning and will compress the truth effect. Soften the wording.

### `fam_coolant__coherent_false`  (stated_confound)

- code `5A0D9C`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four runs are titanium at Kestrel Works on the night shift inside one quarter, so one retooling covers every case at once. The mid-passage line records that "Every shop refitted its machines with a faster insert grade that quarter" and that "every logged run was cut on one of those machines", so the refit reached all four runs. A change of insert grade moves cycle time by itself, and each run is compared against the same run last quarter, when the older grade was still fitted, rather than against a run on the same inserts over the same period. The 4 to 7 minute reductions cannot be separated from the retooling.
- auditor said: A named alternative cause reaches every unit: "Every shop refitted its machines with a faster insert grade that quarter, and every logged run was cut on one of those machines." A faster insert grade shortens cycle time on its own, so the 4-7 minute reductions cannot be attributed to the coolant.
- auditor said: A named alternative cause reaches every unit: "Every shop refitted its machines with a faster insert grade that quarter, and every logged run was cut on one of those machines" - so the 4-7 minute reductions are equally explained by the new insert grade rather than the coolant.

## Worth a look — an auditor named a DIFFERENT flaw

_None._

## TRUE items an auditor thought were broken

These are decoys with no intended flaw. An auditor claiming one means either the item is not as sound as intended — worth checking — or the auditor over-reads. Both matter for interpreting the table above.

- `fam_filter__coherent_true` — answers ['No', 'Yes']
  - auditor said: The particulate figures are partly fabricated: "gaps entered at the permit limit, and a quarter of scheduled runs at each stack returned no reading" means 25 percent of the data is a substituted fixed value rather than a measurement, so the 8-14 percent drops could be driven by how many gaps each quarter had rather than by actual emissions.
- `fam_nestbox__coherent_true` — answers ['No', 'Yes']
  - auditor said: 'Counts cover every installed box, uninspected ones entered as zero, and a fifth of the installed boxes at each plot went uninspected' - so 20 percent of boxes are scored zero by fiat, making the total a function of inspection coverage rather than fledgling production, with no statement that the previous spring's baseline was inspected at the same rate. (The doubled prey abundance is neutralised, since every box 'stood deep inside the reserve, beyond the birds' foraging range'.)
- `fam_seedcoat__coherent_true` — answers ['No', 'Yes']
  - auditor said: The germination figures are partly imputed: "unscored ones counted as nil, and about a fifth of each lot's trays went unscored." A fifth of trays are scored zero without being observed, so a 6-11 point gap against the previous sowing can be produced by a difference in unscored fractions rather than by the coating.
- `fam_tutoring__coherent_true` — answers ['No', 'Yes']
  - auditor said: 'Pass rates cover every enrolled student, non-entrants counted as fails, and about a quarter of each cohort's roll was not entered' - with 25 percent scored as automatic fails, the pass rate tracks the entry rate as much as attainment, and the previous spring term's entry rate is never given, so the 4-8 point rise could come entirely from more students being entered.

## Detection by flaw type

| flaw_mechanism | n | found | missed | too easy |
|---|---|---|---|---|
| broken_chronology | 10 | 10 | 0 | 0 |
| scope_mismatch | 20 | 20 | 0 | 0 |
| stated_confound | 10 | 10 | 0 | 1 |

## Every item

| item | cell | flaw | bucket | answers | explicitness |
|---|---|---|---|---|---|
| `fam_bearing__coherent_false` | coherent_false | stated_confound | found | No,No | 3.5 |
| `fam_bearing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_bearing__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_bearing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_checkout__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_checkout__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_checkout__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_checkout__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_coolant__coherent_false` | coherent_false | stated_confound | too_easy | No,No | 4.0 |
| `fam_coolant__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_coolant__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_coolant__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_driptape__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_driptape__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_driptape__diverse_false` | diverse_false | broken_chronology | found | No,No | 2.0 |
| `fam_driptape__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_fertilizer__coherent_false` | coherent_false | stated_confound | found | No,No | 3.5 |
| `fam_fertilizer__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_fertilizer__diverse_false` | diverse_false | broken_chronology | found | No,No | 2.5 |
| `fam_fertilizer__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_filter__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_filter__coherent_true` | coherent_true |  | false_positive | No,Yes |  |
| `fam_filter__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_filter__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_handwash__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_handwash__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_handwash__diverse_false` | diverse_false | scope_mismatch | found | No,No | 2.5 |
| `fam_handwash__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_inhaler__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_inhaler__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_inhaler__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_inhaler__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_labkit__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_labkit__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_labkit__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_labkit__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_nestbox__coherent_false` | coherent_false | stated_confound | found | No,No | 3.5 |
| `fam_nestbox__coherent_true` | coherent_true |  | false_positive | No,Yes |  |
| `fam_nestbox__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_nestbox__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_onboarding__coherent_false` | coherent_false | stated_confound | found | No,No | 3.5 |
| `fam_onboarding__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_onboarding__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_onboarding__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_physio__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_physio__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_physio__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_physio__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_pricing__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_pricing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_pricing__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_pricing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_reading__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_reading__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_reading__diverse_false` | diverse_false | broken_chronology | found | No,No | 2.0 |
| `fam_reading__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_routing__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_routing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_routing__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_routing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_seedcoat__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_seedcoat__coherent_true` | coherent_true |  | false_positive | No,Yes |  |
| `fam_seedcoat__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_seedcoat__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_sleepapp__coherent_false` | coherent_false | stated_confound | found | No,No | 3.5 |
| `fam_sleepapp__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_sleepapp__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_sleepapp__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_solder__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_solder__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_solder__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_solder__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_timetable__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_timetable__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_timetable__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_timetable__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_tutoring__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_tutoring__coherent_true` | coherent_true |  | false_positive | No,Yes |  |
| `fam_tutoring__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_tutoring__diverse_true` | diverse_true |  | clean | Yes,Yes |  |

## Limitation, stated plainly

The auditors and the item authors are the same model family (DECISIONS.md D-017). This measures whether the flaws are **findable**, not how hard they are for the model under test — an author and an auditor that share priors will agree more than two independent readers would. Re-running the audit with a different model, or handing the drafts to a person, is the way to get a number that means more than this one.
