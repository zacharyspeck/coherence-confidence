# Item audit — blind flaw identification

Step 9. Every item was shown to independent auditors carrying **only** the claim, the passage and the forced question — no cell label, no `ground_truth`, no `confound_note`, and never two items from the same family in one batch. The TRUE items were mixed in as decoys, so an auditor could not know that everything it saw was broken.

## Headline

| | count | of | rate |
|---|---|---|---|
| FALSE items whose intended flaw was found | 40 | 40 | 100% |
| **FALSE items where NO auditor found the flaw — BROKEN** | **0** | 40 | 0% |
| **FALSE items rated as good as stated outright — TOO EASY** | **1** | 40 | 2% |
| TRUE items an auditor invented a flaw for (false positives) | 7 | 40 | 18% |

**How to read the false-positive rate.** Auditors reported a flaw in 7 of 40 items that have nothing wrong with them. Finding a flaw in a FALSE item is only as informative as that rate is low; treat it as the audit's own noise floor, and discount the detection rate above accordingly.

**How to read the 'too easy' count.** The threshold is a mean explicitness of 4.0 out of 5, where 4 means "obvious; the passage all but says it". That bar is arguably unfair to `coherent_false` by construction: the design *requires* the shared confound to be stated in the passage, so an auditor will always see it once they look. Nothing scored 5 ("the passage states the problem outright"). The number to act on is not this count but the asymmetry below.

## The asymmetry that matters

| group | n | mean explicitness (1-5) |
|---|---|---|
| flaw_mechanism `broken_chronology` | 10 | 2.65 |
| flaw_mechanism `scope_mismatch` | 20 | 2.77 |
| flaw_mechanism `stated_confound` | 10 | 3.30 |
| cell `coherent_false` | 20 | **3.10** |
| cell `diverse_false` | 20 | **2.65** |

**Gap: +0.45 points of 5** between the two FALSE cells — OUTSIDE the 0.4 target.

The coherent flaws are still louder. Since the hypothesis predicts AUC(coherent) is LOWER, a louder coherent flaw pushes the primary endpoint the other way — the test is conservative rather than flattering — but a null result could not be distinguished from an effect cancelled by salience.

**Why a residual gap is expected and cannot be driven to zero.** `stated_confound` has to be *stated* in the passage to be a shared confound at all, so it will always be somewhat easier to spot than a date that runs backwards or a population that is quietly narrower. That is D-004, and it is a property of the design rather than of the wording.

**What removes the residual entirely.** The matched-mechanism subset: 10 `scope_mismatch` items in each FALSE cell, drawn from the same 10 families, so the mechanism is identical on both sides and only coherence differs. `src/analyze.py` reports the primary endpoint on that subset directly beneath the full-set version, and carries mean salience beside every AUC plus a logistic model of catch-rate on salience.

On the matched subset the mechanism is `scope_mismatch` on both sides, mean salience 2.77, so the salience difference between conditions there is whatever remains after mechanism is held fixed — reported per condition in the analysis output.

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
- auditor said: A named alternative cause reaches every unit: "Every shop refitted its machines with a faster insert grade that quarter, and every logged run was cut on one of those machines." A faster insert grade directly shortens cycle time, so the 4-7 minute reductions in all four March runs are equally explained by the refit rather than the coolant.

## Worth a look — an auditor named a DIFFERENT flaw

_None._

## TRUE items an auditor thought were broken

These are decoys with no intended flaw. An auditor claiming one means either the item is not as sound as intended — worth checking — or the auditor over-reads. Both matter for interpreting the table above.

- `fam_bearing__diverse_true` — answers ['No', 'No']
  - auditor said: The hours figure is not a measure of the redesigned bearing: "Hours cover every bearing position on each machine, and about a third of each machine's positions had spells on the old bearing." A third of the measured positions were running the old part during the cycle, so the extra 340/290/410/260 hours are an aggregate over mixed old-and-new positions and cannot be attributed to the redesign.
  - auditor said: The outcome measure spans untreated units: "Hours cover every bearing position on each machine, and about a third of each machine's positions had spells on the old bearing." So the 340/290/410/260 extra hours are machine-level totals in which roughly a third of the positions were running the old bearing, and cannot be attributed to the redesign.
- `fam_driptape__coherent_true` — answers ['No', 'Yes']
  - auditor said: The measured population is far wider than the treated one: "Yields cover every hectare in each block fitted with the line, and the line watered about a quarter of each block's hectares intermittently." A 0.4-0.7 t/ha gain averaged over whole blocks, three quarters of whose hectares the line never watered (and the remaining quarter only intermittently), cannot be attributed to the drip line.
- `fam_handwash__diverse_true` — answers ['No', 'Yes']
  - auditor said: The measured population does not match the exposed one: "Rates cover every patient admitted to each ward" while "a quarter of each ward's patients had spells outside the protocol", so a quarter of the patients generating the 6-11 percent drop were not continuously under the handwashing protocol.
- `fam_nestbox__coherent_true` — answers ['Yes', 'No']
  - auditor said: The outcome figures do not isolate the treatment as specified: "Counts cover every installed box, up all season or not, and a fifth of each plot's boxes went up mid-season." A fifth of the boxes in every plot were only in place for part of the breeding season, yet their young are counted as if fully box-exposed, so the fledgling totals mix full and partial exposure.
- `fam_onboarding__diverse_true` — answers ['No', 'No']
  - auditor said: The measured population is not the exposed population: "Retention figures cover every signup, email opened or not, and a third of each cohort's signups never opened it." A third of the people in the 5/3/6/4-point lift never received the treatment at all, so the aggregate rise does not show the email did anything.
  - auditor said: The retention numbers cover a population a third of which never received the treatment: "Retention figures cover every signup, email opened or not, and a third of each cohort's signups never opened it." A 3-6 point cohort-wide lift that includes non-openers cannot be credited to the onboarding email.
- `fam_sleepapp__diverse_true` — answers ['No', 'Yes']
  - auditor said: Exposure is far narrower than measurement: "Readings cover every enrolled user in each cohort" but only "a fifth of each cohort used the app on some nights", so the 3-6 bpm cohort-wide drop is averaged over users who mostly never used the app.
- `fam_tutoring__coherent_true` — answers ['Yes', 'No']
  - auditor said: The measured group is not the treated group: "Pass rates cover every student in each cohort, and about a quarter of each cohort missed at least one block." A quarter of each cohort did not receive the full tutoring, yet all of them are counted in the 4-8 point pass-rate figures, so the reported gain is not attributable to the blocks as delivered.

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
| `fam_bearing__diverse_true` | diverse_true |  | false_positive | No,No |  |
| `fam_checkout__coherent_false` | coherent_false | scope_mismatch | found | No,No | 2.0 |
| `fam_checkout__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_checkout__diverse_false` | diverse_false | scope_mismatch | found | No,No | 2.0 |
| `fam_checkout__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_coolant__coherent_false` | coherent_false | stated_confound | too_easy | No,No | 4.0 |
| `fam_coolant__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_coolant__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_coolant__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_driptape__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_driptape__coherent_true` | coherent_true |  | false_positive | No,Yes |  |
| `fam_driptape__diverse_false` | diverse_false | broken_chronology | found | Yes,No | 2.0 |
| `fam_driptape__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_fertilizer__coherent_false` | coherent_false | stated_confound | found | No,No | 3.5 |
| `fam_fertilizer__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_fertilizer__diverse_false` | diverse_false | broken_chronology | found | No,No | 2.0 |
| `fam_fertilizer__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_filter__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_filter__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_filter__diverse_false` | diverse_false | scope_mismatch | found | No,No | 2.5 |
| `fam_filter__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_handwash__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_handwash__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_handwash__diverse_false` | diverse_false | scope_mismatch | found | No,No | 2.5 |
| `fam_handwash__diverse_true` | diverse_true |  | false_positive | No,Yes |  |
| `fam_inhaler__coherent_false` | coherent_false | stated_confound | found | No,No | 3.5 |
| `fam_inhaler__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_inhaler__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_inhaler__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_labkit__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_labkit__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_labkit__diverse_false` | diverse_false | broken_chronology | found | No,No | 2.5 |
| `fam_labkit__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_nestbox__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_nestbox__coherent_true` | coherent_true |  | false_positive | Yes,No |  |
| `fam_nestbox__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_nestbox__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_onboarding__coherent_false` | coherent_false | stated_confound | found | No,No | 3.5 |
| `fam_onboarding__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_onboarding__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_onboarding__diverse_true` | diverse_true |  | false_positive | No,No |  |
| `fam_physio__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_physio__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_physio__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_physio__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_pricing__coherent_false` | coherent_false | scope_mismatch | found | No,Yes | 3.0 |
| `fam_pricing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_pricing__diverse_false` | diverse_false | scope_mismatch | found | Yes,No | 3.0 |
| `fam_pricing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_reading__coherent_false` | coherent_false | stated_confound | found | No,No | 2.5 |
| `fam_reading__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_reading__diverse_false` | diverse_false | broken_chronology | found | Yes,No | 2.0 |
| `fam_reading__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_routing__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_routing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_routing__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_routing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_seedcoat__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_seedcoat__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_seedcoat__diverse_false` | diverse_false | scope_mismatch | found | No,No | 2.5 |
| `fam_seedcoat__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_sleepapp__coherent_false` | coherent_false | stated_confound | found | No,No | 3.5 |
| `fam_sleepapp__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_sleepapp__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_sleepapp__diverse_true` | diverse_true |  | false_positive | No,Yes |  |
| `fam_solder__coherent_false` | coherent_false | scope_mismatch | found | No,Yes | 3.0 |
| `fam_solder__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_solder__diverse_false` | diverse_false | scope_mismatch | found | No,No | 2.0 |
| `fam_solder__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_timetable__coherent_false` | coherent_false | scope_mismatch | found | Yes,No | 3.0 |
| `fam_timetable__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_timetable__diverse_false` | diverse_false | scope_mismatch | found | No,Yes | 3.0 |
| `fam_timetable__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_tutoring__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_tutoring__coherent_true` | coherent_true |  | false_positive | Yes,No |  |
| `fam_tutoring__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_tutoring__diverse_true` | diverse_true |  | clean | Yes,Yes |  |

## Limitation, stated plainly

The auditors and the item authors are the same model family (DECISIONS.md D-017). This measures whether the flaws are **findable**, not how hard they are for the model under test — an author and an auditor that share priors will agree more than two independent readers would. Re-running the audit with a different model, or handing the drafts to a person, is the way to get a number that means more than this one.
