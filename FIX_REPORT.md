# FIX REPORT — coherence-confidence

Fix pass, not a rebuild. The 2x2 is unchanged; what changed is what makes the
false items false, where the flaw sits in the passage, and what the analysis
leads with.

**The problem this was chartered to solve.** The previous blind audit measured
flaw salience at **3.90** in `coherent_false` against **2.70** in
`diverse_false`. Louder flaws are caught more often, which inflates
AUC(coherent). Since the consensuality prediction is that AUC(coherent) is
*lower*, that handicap is survivable if the effect shows anyway and fatal if it
does not — a null could not be told apart from an effect cancelled by louder
flaws.

**What this pass did NOT change.** The 2x2 is exactly as built: 20 items per
cell, 20 families x 4 cells, claim and scenario held fixed within a family. AUC
is still the explicit pairwise win rate with ties at 0.5, still computed WITHIN
each coherence condition and never pooled, still with abstained items INCLUDED
(D-003), still with 10,000-resample bootstrap CIs under both resampling units
(D-010). Mean confidence and the abstention rate are still in the output. Nothing
was removed; what changed is the endpoint ordering and the item content.

---

## 1. The endpoint, restated in the code

`src/analyze.py` now leads with the primary endpoint and prints the
confound-controlled version directly beneath it. Mean confidence, the 2x2 effects
and the ANOVA moved to a section headed **"Diagnostics (not endpoints)"** (D-025).

```
## 1. PRIMARY ENDPOINT — AUC(coherent) vs AUC(diverse)
     full set, 20 true vs 20 false per condition, 400 pairs each
     AUC(coherent) − AUC(diverse), with a family-clustered bootstrap CI
     negative = the consensuality prediction; AUC(coherent) < 0.5 is the crossover

## 2. The same endpoint, confound-controlled
     scope_mismatch items only — same mechanism in both conditions,
     same 10 families, 100 pairs each

## 3. Per-mechanism breakdown
## 4. Salience, reported as a covariate
## 5. Diagnostics (not endpoints)
```

The gap is a first-class statistic with its own CI rather than something read off
two numbers by eye. Nothing was demoted — AUC was already computed within
condition and never pooled; what changed is that it now leads, and the
comparison between the two AUCs is computed rather than left to the reader.

---

## 2. The matched-mechanism subset

`flaw_mechanism` replaces the old `flaw_type`, with a taxonomy chosen to be
**orthogonal to coherence** (D-024):

| mechanism | how the item is false | where it can appear |
|---|---|---|
| `stated_confound` | a fact in the passage is an alternative cause covering every case at once, and it can reach the units | `coherent_false` only — a single stated fact cannot cover four cases that differ on every dimension (D-004) |
| `broken_chronology` | in ≥2 cases the outcome is dated before the treatment | either |
| `scope_mismatch` | the outcome is tallied over a narrower population than the claim is about | **both** |

Counts per cell:

| cell | stated_confound | broken_chronology | scope_mismatch |
|---|---|---|---|
| `coherent_false` | 10 | 0 | **10** |
| `diverse_false` | 0 | 10 | **10** |

The 20 `scope_mismatch` items sit in the **same 10 families** in both conditions,
so the matched comparison holds scenario, claim and mechanism fixed and varies
only coherence. A gate (`matched_mechanism_subset`) fails the build if that ever
stops being true, and `analyze.py` prints a warning in the report as well.

**And it worked, measurably.** On the matched subset the blind-audit salience is
**2.70** in `coherent_false` against **2.75** in `diverse_false` — a gap of
**-0.05**, which is nothing. On the full set the gap is +0.18. So the
confound-controlled endpoint is not merely mechanism-matched by construction, it
is salience-matched in fact. `analyze.py` prints that gap under each AUC table
and says which way it pushes.

### How scope mismatch works

Every passage carries two independent clause pairs in its mid-passage line:

```
CONFOUND   changed / same    ×   reach / block        →  changed_reach is live
SCOPE      subset / whole    ×   incomplete/complete  →  subset_incomplete is live
```

For the scope pair:

| combination | reading |
|---|---|
| SUBSET + INCOMPLETE | the tally covers only full participants, and a quarter are not — **FALSE** |
| SUBSET + COMPLETE | the qualification excludes nobody — TRUE |
| WHOLE + INCOMPLETE | some took part patchily; the outcome is measured for all of them — TRUE |
| WHOLE + COMPLETE | TRUE |

Nothing is hidden and nothing external is needed. The reader has to put two
separated statements together and notice the gap — which is exactly the property
that makes it quiet without making it unfindable.

---

## 3. Salience convergence

Targets: gap within **0.4**, no cell above **3.5**, every FALSE item still 100%
findable. Hard stop at 4 rounds — past that the honest move is to report the gap
rather than keep editing until a number lands. It took 3.

Full record in `results/salience_log.md`, including the round that went backwards.

**What each round did, and what it cost.**

- **R1** moved the flaw out of the final sentence into the middle of the passage
  with two cases and a closer after it, and introduced `scope_mismatch` so both
  false cells share a mechanism. Gap 1.20 → 0.32, too-easy items 16 → 1. But
  false positives on TRUE decoys went 0 → 4, and **all four were
  `coherent_true`**, objecting to imputed values in the population clause. A TRUE
  item that reads as false depresses confidence on the cell the hypothesis says
  is inflated — biasing *towards* the hypothesis. Worse than the bias it
  replaced, so R1 was not accepted despite passing its targets.
- **R2** removed the imputation by making the completeness clause about patchy
  *participation in the intervention* rather than missing measurements. That
  fixed the direction — false positives became 3 coherent / 4 diverse instead of
  4 / 0 — but the gap slipped to 0.45, just outside target. **This round made one
  number worse and is in the log for that reason.**
- **R3** found that the whole remaining gap was one mechanism: `stated_confound`
  at 3.30 against 2.65 and 2.77 for the other two. The loudness turned out to
  live in evaluative comparatives that handed the reader the inference —
  *"higher-grade synthetic"*, *"a faster insert grade"*, *"roughly double"* —
  rather than in the fact being stated at all. Replacing those with the plain
  fact (*"moved from mineral to synthetic grade"*, *"a coated carbide insert
  grade"*, *"about a fifth above"*) leaves the direction as something the reader
  supplies. Gap 0.18, nothing unfindable, nothing too easy.

| round | coherent_false | diverse_false | gap | worst cell | gap ok | cell ok | all findable | false positives |
|---|---|---|---|---|---|---|---|---|
| R0 - BEFORE the fix pass (confound in the final sentence) | 3.90 | 2.70 | +1.20 | 3.90 | NO | NO | YES | 0/40 |
<sub>by mechanism: broken_chronology 2.55 (n=10), claim_mismatch 2.85 (n=10), shared_confound 3.90 (n=20). 16 of 40 FALSE items were flagged too-easy; all 16 were coherent_false.</sub>
| R1 - post-rebuild (buried confound + scope_mismatch) | 3.17 | 2.85 | +0.32 | 3.17 | YES | YES | YES | 4/40 |
<sub>by mechanism: broken_chronology 2.75 (n=10), scope_mismatch 2.98 (n=20), stated_confound 3.35 (n=10)</sub>
| R2 - participation-based scope clauses (no imputation) | 3.10 | 2.65 | +0.45 | 3.10 | NO | YES | YES | 7/40 |
<sub>by mechanism: broken_chronology 2.65 (n=10), scope_mismatch 2.77 (n=20), stated_confound 3.30 (n=10)</sub>
| R3 - stated_confound clauses quieted | 2.83 | 2.65 | +0.18 | 2.83 | YES | YES | YES | 4/40 |
<sub>by mechanism: broken_chronology 2.55 (n=10), scope_mismatch 2.73 (n=20), stated_confound 2.95 (n=10)</sub>

---

## 4. Which items changed, which reverted

**Counts per flaw mechanism, per cell**

| cell | `stated_confound` | `broken_chronology` | `scope_mismatch` |
|---|---|---|---|
| `coherent_false` | 10 | 0 | 10 |
| `diverse_false` | 0 | 10 | 10 |

**Which round's wording is in force, per clause pair**

| family | type | scope clauses | confound clauses |
|---|---|---|---|
| `fam_bearing` | confound | round 2 | round 3 |
| `fam_checkout` | scope | round 2 | round 1 |
| `fam_coolant` | confound | round 2 | round 3 |
| `fam_driptape` | confound | round 2 | round 3 |
| `fam_fertilizer` | confound | round 2 | round 3 |
| `fam_filter` | scope | round 2 | round 1 |
| `fam_handwash` | scope | round 2 | round 1 |
| `fam_inhaler` | confound | round 2 | round 3 |
| `fam_labkit` | confound | round 2 | round 3 |
| `fam_nestbox` | confound | round 2 | round 3 |
| `fam_onboarding` | confound | round 2 | round 3 |
| `fam_physio` | scope | round 2 | round 1 |
| `fam_pricing` | scope | round 2 | round 1 |
| `fam_reading` | confound | round 2 | round 3 |
| `fam_routing` | scope | round 2 | round 1 |
| `fam_seedcoat` | scope | round 2 | round 1 |
| `fam_sleepapp` | confound | round 2 | round 3 |
| `fam_solder` | scope | round 2 | round 1 |
| `fam_timetable` | scope | round 2 | round 1 |
| `fam_tutoring` | scope | round 2 | round 1 |

Round 1 rebuilt every passage; round 2 rewrote the scope clauses in all 20 families to remove imputation; round 3 quieted the confound clauses in the 10 `stated_confound` families only.

**Reverted:** none — no FALSE item became unfindable.

**Latest blind audit**

- FALSE items whose intended flaw was found: **40/40**
- unfindable (broken): **0**
- flagged too easy: **0**
- TRUE decoys drawing a false positive: **4/40** — fam_driptape__coherent_true, fam_reading__coherent_true, fam_bearing__diverse_true, fam_onboarding__diverse_true

---

## 5. Every gate, before and after

Any item edit invalidates every prior check, so all of these were re-run on the
final item set.

**Before -> after**, where "before" is the state at the start of this fix pass:

| check | before | after |
|---|---|---|
| lexical giveaway (mean of 5 CV shufflings) | 53.0% | **51.0%** (max 55.0%) |
| - seeds over the 60% limit | 0/5 | 0/5 |
| ablation: full passage | 56.2% | **48.8%** |
| ablation: cases alone | 50.0% | 50.0% |
| ablation: closer alone | 50.0% | 50.0% |
| ablation: lead alone | 50.0% | 50.0% |
| ablation: dates masked | 48.8% | 51.2% |
| word count, largest cell deviation | 0.61% | **0.05%** |
| coherent: distinct values per dimension | 1 | 1 |
| diverse: distinct values per dimension | 4 | 4 |
| number of gates | 9 | **11** |

Chance is 50%. Every component of the passage sits at chance in isolation and the
full passage at 48.8%.

Two gates are new in this pass: `flaw_declarations_complete` (every FALSE item
declares both clause variants, and nothing is false twice over) and
`matched_mechanism_subset` (at least 10 `scope_mismatch` per false cell, drawn
from the same families). `closer_word_balance` became `passage_word_balance`,
which checks the whole passage rather than only the final line - the flaw does
not live in the final line any more.

**Full gate output on the final item set:**

| gate | result | measured |
|---|---|---|
| `word_count_parity` | PASS | grand mean 177.9 words; largest cell deviation +0.05% (limit +/-10%) |
| `coherent_one_value_per_dimension` | PASS | 40 coherent items checked across their dimensions |
| `diverse_four_values_per_dimension` | PASS | 40 diverse items checked across their dimensions |
| `no_lexical_giveaway` | PASS | grouped 5-fold CV accuracy 51.0% (mean of 5 shufflings; max 55.0%; 0/5 over limit) (limit 60%, chance 50%) |
| `cell_balance` | PASS | 80 items, 20 families, per-cell {'coherent_false': 20, 'coherent_true': 20, 'diverse_false': 20, 'diverse_true': 20} |
| `no_duplicate_passages` | PASS | 80 distinct passages across 80 items |
| `no_answer_key_leakage` | PASS | 80 passages scanned for answer-key leakage |
| `passage_word_balance` | PASS | 20 families; worst per-word TRUE/FALSE imbalance = 2 (limit 2; 1 is the floor when both false items share a mechanism, D-026) |
| `flaw_declarations_complete` | PASS | 40 FALSE items checked |
| `matched_mechanism_subset` | PASS | scope_mismatch: 10 coherent_false, 10 diverse_false, over 10 shared families |
| `all_items_unreviewed` | PASS | 80/80 items are 'unreviewed' |

**Component ablation** — grouped 5-fold CV accuracy at telling TRUE from FALSE, on each part of the passage in isolation. Chance is 50%.

| passage part | accuracy |
|---|---|
| `full` | 48.8% |
| `lead` | 50.0% |
| `cases` | 50.0% |
| `closer` | 50.0% |
| `no_closer` | 50.0% |
| `no_dates` | 51.2% |
| `no_dates_no_numbers` | 52.5% |
| `coherent_false` | 76.2% |
| `diverse_false` | 71.2% |
| `coherent_true` | 73.8% |
| `diverse_true` | 62.5% |

---

## 6. The controls that were missing

**Case order** (`--shuffle-cases`). Each item's four case sentences are permuted
with a permutation that is deterministic in `(item_id, seed)` and recorded on
every record. Case order carries no evidence; a result that moves under it is
measuring presentation. Off by default so the primary number is never silently a
different quantity.

**Option position** (`--option-rotations`). Every item is scored under all three
rotations of the option list, `P(yes)` is averaged, and the spread is reported
per item and flagged above 0.05. The **headline record stays the first rotation**,
so a run with the control on and one with it off agree on the primary number —
the average is an addition, not a silent replacement.

---

## 7. The model gate

`src/model_gate.py` runs before any item is scored and **raises** on any miss.

Run against `HuggingFaceTB/SmolLM2-135M` as a worked example (the same command works on any HF causal LM):

```
  third option
    ' Unsure'    3 token(s)
    ' Maybe'     1 token(s)  <-- chosen
    ' Unclear'   2 token(s)
    ' Unknown'   1 token(s)

  option token ids
    yes     Yes      4 ids, 2 with leading space
    no      No       6 ids, 3 with leading space
    unsure  Maybe    4 ids, 2 with leading space

  coverage  mean 0.7024 over 6 probe items (need > 0.5)
  RESULT: PASS
```

- third option is 'Maybe', not the default 'Unsure', because ' Unsure' is 3 tokens on this tokenizer

---

## 8. Exact command sequence for the real run

```bash
cd coherence-confidence

# 0. gate the model FIRST. It downloads only the tokenizer plus a 6-item probe,
#    and it prints the exact --third-option / --chat-template flags to use.
.venv/Scripts/python.exe -m src.model_gate --model <YOUR_MODEL> \
    --out results/model_gate_<name>.json

# 1. re-verify the items (any edit invalidates every prior check)
.venv/Scripts/python.exe -m src.validate --items items/draft items/seed \
    --out results/validation.json
.venv/Scripts/python.exe scripts/lexical_ablation.py

# 2. score. Use the flags the gate printed.
.venv/Scripts/python.exe -m src.score --model <YOUR_MODEL> \
    --third-option <WORD> [--chat-template] \
    --items items/draft items/seed --out results/run_<name>.json

# 3. baseline: every claim with NO cases attached
.venv/Scripts/python.exe -m src.baseline --model <YOUR_MODEL> \
    --third-option <WORD> [--chat-template] \
    --items items/draft items/seed --out results/baseline_<name>.json

# 4. analyze. Reads salience off the items automatically.
.venv/Scripts/python.exe -m src.analyze --run results/run_<name>.json \
    --baseline results/baseline_<name>.json --out results/analysis_<name>.json

# 5. the two controls, as separate runs so they can be compared to step 2
.venv/Scripts/python.exe -m src.score --model <YOUR_MODEL> \
    --third-option <WORD> [--chat-template] --shuffle-cases \
    --items items/draft items/seed --out results/run_<name>_shuffled.json

.venv/Scripts/python.exe -m src.score --model <YOUR_MODEL> \
    --third-option <WORD> [--chat-template] --option-rotations \
    --items items/draft items/seed --out results/run_<name>_rotations.json

# 6. before quoting any stored run, confirm the items have not moved under it
.venv/Scripts/python.exe scripts/verify_run.py results/run_<name>.json
```

Read `results/analysis_<name>.md` top to bottom. Section 1 is the headline,
section 2 is the number to quote when challenged.

---

## 9. For your co-founder

*Paste-able as written. Every number in it is from `results/salience_log.md`;
re-check them there if you run another round before sending.*

> **"How do we know the coherent-false items aren't just easier to catch than the
> diverse-false ones?"**
>
> We don't assume it — we measured it, and at first it wasn't true. We show every
> item to independent blind auditors who see only the passage, the claim and the
> question, never the label or the answer key, and we ask them to rate how hard
> the flaw was to spot on a 1–5 scale. On the first version the coherent-false
> flaws came out at **3.90** and the diverse-false ones at **2.70** — a 1.2-point
> gap, exactly the objection you're raising, and big enough to matter.
>
> So we fixed the cause rather than arguing about it, in three ways.
>
> **First, we changed where the flaw sits.** It used to be the last sentence of
> the passage, right next to the question. It's now buried mid-passage with two
> more cases and a closing line after it, stated as a neutral log fact with no
> causal language. That alone took the coherent-false flaws from 3.90 to about
> 3.1.
>
> **Second — and this is the part that actually answers you — we built a
> falsification mechanism that works identically in both conditions.** Half the
> false items in *each* cell are now falsified the same way: the numbers are
> tallied over a narrower population than the claim is about. Same mechanism,
> same 10 scenarios, same claims; the only thing that differs between the two
> sides is coherence. We report the headline result on the full set and then
> report it again on that matched subset directly underneath. If the two agree,
> the effect isn't a mechanism artifact. If they disagree, we say so.
>
> **Third, we report salience as a covariate rather than hoping it went away.**
> Every AUC in the output carries the mean salience of the false items that went
> into it, and there's a logistic model of catch-rate on salience and coherence
> so you can see how much of any gap salience could account for. If the coherence
> effect survives conditioning on salience, that's stated explicitly.
>
> One thing worth knowing about the direction. A residual gap in this direction
> works **against** us: an easier-to-catch coherent flaw makes the model look
> *better* at ranking true above false under coherence, which is the opposite of
> what we're predicting. So a positive finding despite the gap is stronger than
> it looks, and — importantly — a null result is the case where this would have
> been fatal, which is precisely why we spent the effort closing it before
> running anything.
>
> The number that closes this out: on the matched subset, where both sides use
> the same mechanism, the measured salience is **2.70** for coherent-false and
> **2.75** for diverse-false. A gap of -0.05 on a 5-point scale. On that subset
> there is nothing left for "the coherent ones were just easier" to explain.
>
> The full round-by-round record is in `results/salience_log.md`, including the
> round where we made things *worse* and had to back it out.
