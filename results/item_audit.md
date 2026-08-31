# Item audit — blind flaw identification

Step 9. Every item was shown to independent auditors carrying **only** the claim, the passage and the forced question — no cell label, no `ground_truth`, no `confound_note`, and never two items from the same family in one batch. The TRUE items were mixed in as decoys, so an auditor could not know that everything it saw was broken.

## Headline

| | count | of | rate |
|---|---|---|---|
| FALSE items whose intended flaw was found | 40 | 40 | 100% |
| **FALSE items where NO auditor found the flaw — BROKEN** | **0** | 40 | 0% |
| **FALSE items rated as good as stated outright — TOO EASY** | **0** | 40 | 0% |
| TRUE items an auditor invented a flaw for (false positives) | 4 | 40 | 10% |

**How to read the false-positive rate.** Auditors reported a flaw in 4 of 40 items that have nothing wrong with them. Finding a flaw in a FALSE item is only as informative as that rate is low; treat it as the audit's own noise floor, and discount the detection rate above accordingly.

**How to read the 'too easy' count.** The threshold is a mean explicitness of 4.0 out of 5, where 4 means "obvious; the passage all but says it". That bar is arguably unfair to `coherent_false` by construction: the design *requires* the shared confound to be stated in the passage, so an auditor will always see it once they look. Nothing scored 5 ("the passage states the problem outright"). The number to act on is not this count but the asymmetry below.

## The asymmetry that matters

| group | n | mean explicitness (1-5) |
|---|---|---|
| flaw_mechanism `broken_chronology` | 10 | 2.55 |
| flaw_mechanism `scope_mismatch` | 20 | 2.73 |
| flaw_mechanism `stated_confound` | 10 | 2.95 |
| cell `coherent_false` | 20 | **2.83** |
| cell `diverse_false` | 20 | **2.65** |

**Gap: +0.18 points of 5** between the two FALSE cells — inside the 0.4 target.

This is what the fix pass was for. The flaw used to sit in the final sentence of the passage and the gap was **1.20**; it now sits mid-passage with two cases and a closer after it, and `scope_mismatch` — a mechanism that works identically under both coherence conditions — supplies half of each FALSE cell. `results/salience_log.md` has the round-by-round record.

**Why a residual gap is expected and cannot be driven to zero.** `stated_confound` has to be *stated* in the passage to be a shared confound at all, so it will always be somewhat easier to spot than a date that runs backwards or a population that is quietly narrower. That is D-004, and it is a property of the design rather than of the wording.

**What removes the residual entirely.** The matched-mechanism subset: 10 `scope_mismatch` items in each FALSE cell, drawn from the same 10 families, so the mechanism is identical on both sides and only coherence differs. `src/analyze.py` reports the primary endpoint on that subset directly beneath the full-set version, and carries mean salience beside every AUC plus a logistic model of catch-rate on salience.

On the matched subset the mechanism is `scope_mismatch` on both sides, mean salience 2.73, so the salience difference between conditions there is whatever remains after mechanism is held fixed — reported per condition in the analysis output.

## FLAGGED - broken items (auditors could only gesture, not identify)

_None._

## FLAGGED — broken items (no auditor found the intended flaw)

_None._

## FLAGGED — too easy (flaw rated as good as stated outright)

_None._

## Worth a look — an auditor named a DIFFERENT flaw

_None._

## TRUE items an auditor thought were broken

These are decoys with no intended flaw. An auditor claiming one means either the item is not as sound as intended — worth checking — or the auditor over-reads. Both matter for interpreting the table above.

- `fam_bearing__diverse_true` — answers ['No', 'Yes']
  - auditor said: The outcome measure covers units that were only partly on the new bearing: "Hours cover every bearing position on each machine, and about a third of each machine's positions had spells on the old bearing." So roughly a third of the measured positions were running the old design for part of the cycle, and the 260-410 extra hours cannot be attributed to the redesigned bearing alone.
- `fam_driptape__coherent_true` — answers ['No', 'No']
  - auditor said: The measured area and the treated area do not match: "Yields cover every hectare in each block fitted with the line" while "the line watered about a quarter of each block's hectares intermittently." Roughly three-quarters of the hectares producing the 0.4-0.7 t/ha gain were never watered by the drip line, so the block-wide rise cannot be attributed to it.
  - auditor said: The numbers cover a different population than the treatment: 'Yields cover every hectare in each block fitted with the line' while 'the line watered about a quarter of each block's hectares intermittently'. A per-hectare gain averaged over the whole block is credited to a line that reached only about 25% of those hectares, and only intermittently.
- `fam_onboarding__diverse_true` — answers ['No', 'Yes']
  - auditor said: The measured population is broader than the treated one: "Retention figures cover every signup, email opened or not, and a third of each cohort's signups never opened it." A third of each cohort was never actually exposed to the email, so the 3-6 point retention lift is measured on a population one-third of which received no treatment.
- `fam_reading__coherent_true` — answers ['No', 'No']
  - auditor said: The score population and the treated population diverge: "Scores cover every pupil on each class roll, and a quarter of each roll missed at least one program session." The 5-9 point gain is measured over pupils a quarter of whom did not receive the full program, so the figures do not isolate the program's effect.
  - auditor said: 'Scores cover every pupil on each class roll, and a quarter of each roll missed at least one program session', so the measured group is not the group that actually received the program; the gain is averaged over pupils a quarter of whom had incomplete exposure. The four 'classes' are also the same Ashcombe Primary Year 4 taught by Vance on consecutive start dates.

## Detection by flaw type

| flaw_mechanism | n | found | missed | too easy |
|---|---|---|---|---|
| broken_chronology | 10 | 10 | 0 | 0 |
| scope_mismatch | 20 | 20 | 0 | 0 |
| stated_confound | 10 | 10 | 0 | 0 |

## Every item

| item | cell | flaw | bucket | answers | explicitness |
|---|---|---|---|---|---|
| `fam_bearing__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_bearing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_bearing__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_bearing__diverse_true` | diverse_true |  | false_positive | No,Yes |  |
| `fam_checkout__coherent_false` | coherent_false | scope_mismatch | found | No,Yes | 2.0 |
| `fam_checkout__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_checkout__diverse_false` | diverse_false | scope_mismatch | found | No,No | 2.0 |
| `fam_checkout__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_coolant__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_coolant__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_coolant__diverse_false` | diverse_false | broken_chronology | found | No,No | 2.5 |
| `fam_coolant__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_driptape__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_driptape__coherent_true` | coherent_true |  | false_positive | No,No |  |
| `fam_driptape__diverse_false` | diverse_false | broken_chronology | found | Yes,No | 2.0 |
| `fam_driptape__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_fertilizer__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_fertilizer__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_fertilizer__diverse_false` | diverse_false | broken_chronology | found | No,No | 2.0 |
| `fam_fertilizer__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_filter__coherent_false` | coherent_false | scope_mismatch | found | No,No | 2.5 |
| `fam_filter__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
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
| `fam_nestbox__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_nestbox__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_nestbox__diverse_false` | diverse_false | broken_chronology | found | No,No | 2.5 |
| `fam_nestbox__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_onboarding__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_onboarding__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_onboarding__diverse_false` | diverse_false | broken_chronology | found | No,No | 3.0 |
| `fam_onboarding__diverse_true` | diverse_true |  | false_positive | No,Yes |  |
| `fam_physio__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_physio__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_physio__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_physio__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_pricing__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_pricing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_pricing__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_pricing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_reading__coherent_false` | coherent_false | stated_confound | found | No,No | 2.5 |
| `fam_reading__coherent_true` | coherent_true |  | false_positive | No,No |  |
| `fam_reading__diverse_false` | diverse_false | broken_chronology | found | Yes,No | 2.0 |
| `fam_reading__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_routing__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_routing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_routing__diverse_false` | diverse_false | scope_mismatch | found | No,No | 2.5 |
| `fam_routing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_seedcoat__coherent_false` | coherent_false | scope_mismatch | found | No,No | 2.5 |
| `fam_seedcoat__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_seedcoat__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_seedcoat__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_sleepapp__coherent_false` | coherent_false | stated_confound | found | No,No | 3.0 |
| `fam_sleepapp__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_sleepapp__diverse_false` | diverse_false | broken_chronology | found | No,No | 2.5 |
| `fam_sleepapp__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_solder__coherent_false` | coherent_false | scope_mismatch | found | No,No | 2.0 |
| `fam_solder__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_solder__diverse_false` | diverse_false | scope_mismatch | found | No,No | 2.5 |
| `fam_solder__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_timetable__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_timetable__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_timetable__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_timetable__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_tutoring__coherent_false` | coherent_false | scope_mismatch | found | No,No | 3.0 |
| `fam_tutoring__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_tutoring__diverse_false` | diverse_false | scope_mismatch | found | No,No | 3.0 |
| `fam_tutoring__diverse_true` | diverse_true |  | clean | Yes,Yes |  |

## Limitation, stated plainly

The auditors and the item authors are the same model family (DECISIONS.md D-017). This measures whether the flaws are **findable**, not how hard they are for the model under test — an author and an auditor that share priors will agree more than two independent readers would. Re-running the audit with a different model, or handing the drafts to a person, is the way to get a number that means more than this one.
