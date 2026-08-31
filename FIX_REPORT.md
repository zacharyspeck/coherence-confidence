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

Full round-by-round record in `results/salience_log.md`.

<!--SALIENCE_TABLE-->

---

## 4. Which items changed, which reverted

<!--CHANGE_TABLE-->

---

## 5. Every gate, before and after

<!--GATE_TABLE-->

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

<!--GATE_OUTPUT-->

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

<!--COFOUNDER-->
