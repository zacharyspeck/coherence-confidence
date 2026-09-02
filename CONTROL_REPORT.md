# CONTROL REPORT — decorative diversity

One control, built to close the last standing version of the objection.

**The gap it closes.** Diverse passages carry far more surface variety than
coherent ones — four countries, four devices, four months against one of each.
More entities to track. A confidence difference could come from parse load rather
than from the evidence being evidentially independent. Nothing in the design
addressed that: the matched-mechanism subset holds the *flaw* fixed, not the
reading difficulty.

**What was built.** 20 new items — 10 true, 10 false — in the same 10
`scope_mismatch` families as the matched subset, so claim, scenario and
falsification mechanism are all held fixed. In them:

- every dimension that **bears on the claim** — region, season, population,
  treatment context — is **identical across all four cases**, exactly as in a
  coherent item;
- each case carries four **distinct decorations** — a log serial, a clock time
  inside the same day, a terminal, a desk — which rule nothing out and therefore
  confer no evidential independence at all.

Entity count and surface busyness match the diverse cells. Evidential structure
matches the coherent cells.

---

## 1. What it decides

`src/analyze.py` states the answer in one sentence in section 1 of every
analysis, phrased to whichever of three things the data says:

| decorative AUC sits with | the report says |
|---|---|
| **coherent** | the effect is about evidential independence, not parse load |
| **diverse** | the effect is surface complexity and the story is wrong |
| **neither** | this run does not separate the two |

Section 3 carries the table, with mean distinct-entities and mean condition-values
printed beside each AUC so the control can be seen working rather than taken on
trust. The comparison runs **inside the 10 control families only** — same
scenarios, same claims, same mechanism — at 10 true vs 10 false, 100 pairs per
level.

Two gaps are reported with family-clustered bootstrap CIs: `decorative − coherent`
and `decorative − diverse`. Near zero on the first means the control behaves like
a coherent item.

---

## 2. Entity and complexity match

<!--MATCH_TABLE-->
| metric | coherent | diverse | **decorative** | decorative vs diverse |
|---|---|---|---|---|
| distinct tokens | 78.9 | 96.9 | **98.5** | +1.6% |
| distinct entities | 13.7 | 26.1 | **25.2** | -3.3% |
| entity tokens (with repeats) | 30.3 | 30.4 | **41.7** | +37.4% |
| type-token ratio | 0.433 | 0.532 | **0.496** | -6.8% |
| distinct CONDITION values | 4.0 | 16.0 | **4.0** | -75.0% |
| distinct DECORATION values | 0.0 | 0.0 | **16.0** | — |
| mean word count | 181.9 | 182.1 | **198.6** | — |

- `coherent` mean word count 181.9, -1.83% from the grand mean of 185.3 (limit ±10%)
- `diverse` mean word count 182.1, -1.75% from the grand mean of 185.3 (limit ±10%)
- `decorative` mean word count 198.6, +7.18% from the grand mean of 185.3 (limit ±10%)
<!--/MATCH_TABLE-->

Read **distinct entities** against **distinct condition values**. The first is
what the objection is about — how many different things a reader has to track —
and decorative sits with diverse. The second is what the coherent/diverse
manipulation actually changes, and decorative sits with coherent, at a quarter of
diverse. Both at once is what makes this a control rather than a third condition.

Two rows deserve a caveat rather than a claim. **Entity tokens with repeats** is
*higher* in decorative than in either other level: the decorations add mentions on
top of a passage that already repeats one site name four times. Distinct entities
is the measure the objection is actually about — you track distinct things, not
mentions — which is why the gate uses it, but the difference is real and is here
rather than buried. **Type-token ratio** sits 8% below diverse, inside the 10%
tolerance but not by much, because decorative passages are slightly longer for the
same vocabulary.

Gated by `control_surface_match`, which fails the build if either half drifts:
the surface must stay within 10% of diverse **and** the condition variety must
equal coherent exactly and stay below diverse. Half a control is not a control.

---

## 3. Item counts

<!--COUNTS_TABLE-->
| cell | n | flaw mechanism |
|---|---|---|
| `coherent_true` | 20 | — |
| `coherent_false` | 20 | scope_mismatch, stated_confound |
| `diverse_true` | 20 | — |
| `diverse_false` | 20 | broken_chronology, scope_mismatch |
| `decorative_true` | 10 | — |
| `decorative_false` | 10 | scope_mismatch |

**100 items in 20 families.**
<!--/COUNTS_TABLE-->

The control arm is deliberately half the size of a core cell. It exists in the 10
`scope_mismatch` families so that it is drawn from the same scenarios as the
matched subset — that comparability is worth more than a larger n on a control.
It rests on 100 pairs against the primary endpoint's 400, and it is a control,
not a second primary endpoint.

---

## 4. Salience of the new false items

The decorative items went through the same blind audit as everything else: two
independent passes, auditors seeing only the claim, the passage and the question,
50 TRUE items mixed in as decoys, and reported flaws matched against the answer
key by separate strict judges.

<!--SALIENCE_TABLE-->
| false cell | n | mean salience | flaws found | too easy |
|---|---|---|---|---|
| `coherent_false` | 20 | **2.83** | 20/20 | 0 |
| `diverse_false` | 20 | **2.55** | 20/20 | 0 |
| `decorative_false` | 10 | **2.65** | 10/10 | 0 |

- `coherent_false` − `diverse_false` = **+0.28**
- `decorative_false` − `coherent_false` = **-0.18**
- `decorative_false` − `diverse_false` = **+0.10**

Largest pairwise gap **0.28**, against a target of 0.40. No cell exceeds 3.5 (worst is 2.83).

False positives on TRUE decoys: **5/50** — coherent_true 4, decorative_true 1
<!--/SALIENCE_TABLE-->

The control had to clear the same bar as the other two false cells, and it does:
every one of its flaws was found, none was rated as good as stated outright, and
its salience sits between the other two rather than off to one side. If it had
been quieter or louder, a decorative-vs-anything AUC difference could have been
flaw loudness rather than the thing being controlled for.

Round-by-round record in `results/salience_log.md`.

---

## 5. Every gate

Item edits invalidate every prior check, so all of these were re-run on the
100-item set.

<!--GATE_TABLE-->
| gate | before (80 items, `b5f4ac5`) | after (100 items) | measured now |
|---|---|---|---|
| `word_count_parity` | PASS | PASS | grand mean 185.3 words; largest cell deviation +7.66% (limit +/-10%) |
| `coherent_one_value_per_dimension` | PASS | PASS | 60 coherent + decorative items checked across their condition dimensions |
| `diverse_four_values_per_dimension` | PASS | PASS | 40 diverse items checked across their dimensions |
| `no_lexical_giveaway` | PASS | PASS | grouped 5-fold CV accuracy 49.3% (mean of 5 shufflings; max 51.1%; 0/5 over limit) (limit 60%, chance 50%) |
| `cell_balance` | PASS | PASS | 100 items, 20 families (10 with a control arm), per-cell {'coherent_false': 20, 'coherent_true': 20, 'diverse_false': 20, 'diverse_true': 20, 'decorative_false': 10, 'decorative_true': 10} |
| `no_duplicate_passages` | PASS | PASS | 100 distinct passages across 100 items |
| `no_answer_key_leakage` | PASS | PASS | 100 passages scanned for answer-key leakage |
| `passage_word_balance` | PASS *(redefined)* | PASS | 1015 distinct words; flagged when \|T-F\| > 4, the split exceeds 40%, and the word spans >= 3 families. 0 flagged; worst qualifying split 17% |
| `flaw_declarations_complete` | PASS | PASS | 50 FALSE items checked |
| `matched_mechanism_subset` | PASS | PASS | scope_mismatch: 10 coherent_false, 10 diverse_false, over 10 shared families |
| `no_duplicate_flaw_phrasing` | — *(new)* | PASS | 50 FALSE items, 1185 cross-family pairs; worst similarity 0.62 (limit 0.7), no shared 8-grams; 40 within-family pairs exempt (matched mechanism, D-024) |
| `control_surface_match` | — *(new)* | PASS | 20 decorative items; largest surface gap vs diverse +3.3% (limit +/-10%); condition values 4.0 vs coherent 4.0, diverse 16.0 |
| `all_items_unreviewed` | PASS | PASS | 100/100 items are 'unreviewed' |

**Component ablation** — grouped 5-fold CV accuracy at telling TRUE from FALSE on each part of the passage in isolation. Chance is 50%.

| passage part | accuracy |
|---|---|
| `full` | 49.9% |
| `lead` | 50.0% |
| `cases` | 50.8% |
| `closer` | 50.0% |
| `no_closer` | 49.8% |
| `no_dates` | 46.6% |
| `no_dates_no_numbers` | 46.6% |
<!--/GATE_TABLE-->

### What changed in the gates

Two gates are new or reframed in this pass.

**`control_surface_match`** is new. It is the gate that makes the control mean
anything, and it checks both halves — surface within 10% of diverse, condition
variety equal to coherent and strictly below diverse.

**`passage_word_balance` moved from per-family to global.** Adding a 6-item family
broke it, and the failure was informative: it flagged `'in'` at 3T/0F inside one
family while that word sat at **100T/97F globally** — a 1.5% split and no signal
at all. The gate is a proxy for `no_lexical_giveaway`, which trains on 16 families
and tests on 4, so what a classifier can exploit is the *global* association; a
word leaning TRUE in one family and FALSE in another cancels. Per-family balance
is also arithmetically unreachable once a family holds six items. It now flags a
word only when it is lopsided absolutely, lopsided relatively, **and** present in
at least three families — a word confined to one family cannot cross a grouped
fold boundary in either direction.

Reframed, it immediately found something real that the control arm had
introduced: `'any'` at 14T/4F across 10 families and `'no'` at 17T/7F across 13,
because all ten scope families used the same *"and no X in any Y"* frame for the
COMPLETE clause and the new arm made COMPLETE lean TRUE. Fixed by giving each of
the ten a different construction — dispersing beats mirroring, which would have
concentrated the lean into one word at 15T/0F. The gate was checked both ways: it
fails on the pre-fix wording and passes now.

---

## 6. Conditioning on surface complexity

Every item carries a `surface_complexity` covariate — token counts, distinct
tokens, type-token ratio, entity counts, and condition and decoration variety. It
is a pure function of the passage, recomputed and checked at load exactly like
`word_count`, so it cannot drift.

`analyze.py` fits three nested logistic models of catch-rate over the FALSE items
in the core cells:

```
caught ~ coherent
caught ~ coherent + salience
caught ~ coherent + salience + surface complexity
```

and reports the coherence coefficient down that path, so *"does the effect survive
conditioning"* is directly readable rather than inferred. It answers both standing
objections at once — flaw loudness and parse load.

The decorative arm is the stronger answer of the two, and the report says so: it
holds surface complexity fixed **by construction** rather than adjusting for it
after the fact. The regression is the supporting check.

It also declines to fit rather than returning a meaningless number. On the
stand-in run below the model never catches anything, so the outcome is constant
and a logistic fit is undefined; the analysis says exactly that instead of
printing a coefficient.

---

## 7. The pipeline run — plumbing, not evidence

The whole 100-item set was scored end to end on **SmolLM2-135M**, a 135-million
parameter base model. This is a wiring check. It is **not** evidence about the
hypothesis and nothing below should be read as a result.

Why it can't be: the model answers *yes* to essentially everything. Mean P(yes)
sits at 0.67–0.70 in **all six cells**, the spread between cells is under 0.03,
abstention is 0.000 everywhere, and it catches **0 of 40** false items. A model
that always says yes has no confidence-accuracy relationship to measure, so every
AUC here is noise around 0.5 and every CI spans it.

What the run does establish is that the machinery works on real items:

| level (10 control families) | AUC | 95% CI (family) | distinct entities | condition values |
|---|---|---|---|---|
| coherent | 0.5600 | [0.4062, 0.7273] | 13.2 | 4.0 |
| diverse | 0.4300 | [0.2917, 0.5432] | 24.7 | 16.0 |
| **decorative** | **0.5000** | [0.3889, 0.6111] | **25.2** | **4.0** |

- `decorative − coherent` = **−0.060**, CI [−0.198, +0.078]
- `decorative − diverse` = **+0.070**, CI [−0.130, +0.299]

The control lands almost exactly between the two, and the analysis says so in
section 1, in the words it was going to use either way:

> *The decorative control sits between the coherent and diverse cells (AUC 0.500
> against 0.560 coherent and 0.430 diverse), so this run does not separate
> evidential independence from surface complexity.*

That is the third of the three sentences in the table in section 1 above, and
seeing it fire is the point of running this. The design commits to a reading in
advance; the stand-in model produces the one case where the honest answer is
*this doesn't tell you*, and the report gives that answer rather than picking
whichever neighbour is marginally closer. The entity and condition columns beside
the AUCs confirm the control held while it did — 25.2 entities against diverse's
24.7, 4.0 condition values against coherent's 4.0.

The primary endpoint on this run is **+0.040**, CI [−0.075, +0.168] — sign
opposite to the consensuality prediction, interval spanning zero. Again: the model
is answering yes to everything. Run it on a model that discriminates before
reading anything into any of these numbers.

---

## 8. For your co-founder

> **"How do you know the diverse items aren't just harder to read?"**
>
> We built the item that separates those two things. In the diverse cells the four
> cases differ on dimensions that bear on the claim — four countries, four
> devices, four months — and that variety is doing two jobs at once: it makes the
> evidence more independent, and it makes the passage busier to read. Your
> objection is that we can't tell which one is moving confidence. That's fair, and
> nothing we'd built until now separated them.
>
> So there is now a third arm. In it, every dimension that bears on the claim is
> **identical across all four cases**, exactly as in a coherent item — same
> region, same season, same population. But each case carries four *different*
> log serials, clock times, terminals and desks. Those rule nothing out. Knowing
> that a reading was taken at 0915 on terminal T7 rather than 1040 on T8 tells
> you nothing about whether the result generalises. The passage is as busy as a
> diverse one and as evidentially dependent as a coherent one.
>
> The match is measured, not asserted, and it is gated. Distinct entities land
> within a few percent of the diverse cells; distinct condition values land
> exactly on the coherent cells at a quarter of diverse. The build fails if either
> half drifts.
>
> So the reading is clean. If confidence on the decorative items tracks the
> **coherent** items, the effect is about evidential independence and surface
> complexity isn't driving it. If it tracks the **diverse** items, the effect is
> parse load and our story is wrong — and we'd report that, because we wrote the
> sentence that says it into the analysis before we ran anything. The analysis
> prints whichever of those three it is in section 1, in one sentence, and puts
> the entity counts beside the AUCs so you can check the control was actually
> holding.
>
> One thing to hold us to: the control lives in 10 of the 20 families, so it rests
> on 100 pairs against the primary endpoint's 400. It is a control, not a second
> headline. And we also report the regression version — the coherence effect
> conditioned on a surface-complexity covariate — but the arm is the better
> answer, because it holds the thing fixed instead of adjusting for it afterwards.
>
> These 20 items also went through the same blind audit as everything else, and
> had to clear the same bar: every flaw findable, none obvious, and salience
> sitting with the other two false cells rather than off to one side. It does.
>
> To be clear about where we are: the only model we've run end to end so far is a
> 135M base model that answers *yes* to almost everything — mean confidence within
> 0.03 across all six cells, zero abstentions, zero of forty flaws caught. It
> confirms the pipeline runs on the real items and that the verdict logic fires;
> it tells us nothing about the hypothesis, and on it the control lands exactly
> between the two, which the report states as "this run does not separate them."
> The items are built and gated. The measurement is the next step, not a
> completed one.

---

## 9. Commands

```bash
# rebuild the control arm after any change to the core items
.venv/Scripts/python.exe scripts/assemble_passages.py     # the 2x2
.venv/Scripts/python.exe scripts/build_decorative.py      # the control arm
.venv/Scripts/python.exe scripts/apply_complexity.py      # refresh the covariate

# re-verify
.venv/Scripts/python.exe -m src.validate --items items/draft items/seed
.venv/Scripts/python.exe scripts/lexical_ablation.py

# re-audit (the control arm must keep clearing the salience bar)
.venv/Scripts/python.exe scripts/make_audit_batches.py --round 0
.venv/Scripts/python.exe scripts/make_audit_batches.py --round 1
#   ... run the auditors, then:
.venv/Scripts/python.exe scripts/score_audit.py --round-label "R5 - <what changed>"
.venv/Scripts/python.exe scripts/apply_salience.py

# rebuild this report from the artifacts
.venv/Scripts/python.exe scripts/build_control_report.py
```

Run `assemble_passages.py` **before** `build_decorative.py` — the control arm is
built from `coherent_true`, so it has to be regenerated after the core items move.
