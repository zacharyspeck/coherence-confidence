# READER REPORT — measuring findability instead of assuming it

A 10-item human spot-check caught **1 of 5** flaws and read one TRUE item as
FALSE. The blind audit reported **94%** per auditor across the 50 FALSE items.
Both numbers were right. They were measuring different things, and only one of
them was the thing the item set needed to be true.

This pass builds the other measurement, then fixes what it exposes.

---

## 1. Why the old number was never findability

The existing audit hands its auditor this instruction:

> For each one, answer the question and say whether you can identify a specific
> reason the evidence fails to establish the claim. Some of these items have
> nothing wrong with them.

That is a **hunter**. It announces that flaws exist, asks for one by name, and
gives a field to write it in. Near-ceiling performance is what that instruction
is built to produce, and two things follow:

- **`salience` is conditional on discovery.** It is rated by someone who already
  found the flaw, so it measures how explicit the thing looks once you are
  looking at it — not how often anyone looks.
- **As a covariate it is nearly empty.** The whole 100-item set spanned 2.0 to
  3.0. Conditioning on a variable with almost no variance cannot move a
  coefficient, so the "are the coherent flaws just louder?" objection was being
  answered with a number that could not have detected the problem.

There was also a smaller overstatement worth naming: the headline "50/50 found,
0 missed" used an **any-of-two-auditors** rule. Both auditors found the flaw on
44 items, exactly one found it on 6, neither on 0. The honest per-reader figure
was 94%, not 100%.

But that correction is nowhere near large enough. Against a 94% per-reader rate,
catching 1 of 5 has probability **6×10⁻⁵**. The hunter and a plain reader are
not sampling the same process, and no adjustment to the hunter's arithmetic was
going to close that.

## 2. What the reader audit does

`src/audit_reader.py`. A reader sees **exactly `render_prompt(item)`** — the same
instruction, claim, passage and three options the scored model gets — and
nothing else. No mention that a flaw might exist. No request for a description.
No explicitness field, because the scored model has nowhere to put one either.
The only thing recorded is which of Yes / No / Unsure it picks.

Three independent runs per item, regrouped each run, giving a rate rather than a
bit:

| measure | definition |
|---|---|
| `reader_catch_rate` | fraction answering **No** on a FALSE item |
| `reader_false_positive_rate` | fraction answering **No** on a TRUE item |
| `abstention_rate` | fraction answering **Unsure** |

Blindness works as in the hunter audit and for the same reasons: opaque per-item
codes, TRUE items mixed throughout, the key written to a separate tree, and no
two items from one family in a batch. The batch builder additionally **refuses to
emit a file** containing a cell name or any of the hunter's cue words, so the
reader cannot be told to look by accident.

`reader_catch_rate` replaces `salience` as the covariate in `analyze.py`. Hunter
salience stays in the output, labelled as what it is.

### Hunter against reader, on the same items

<!--HUNTER_VS_READER-->
<!--/HUNTER_VS_READER-->

The gap between the last two columns of the first table is the whole point of
this pass.

## 3. Triage, before and after

<!--TRIAGE-->
<!--/TRIAGE-->

## 4. What the rewrite changed

### TRUE items that read as false (D-034)

The confound and the fact that disarms it shared one sentence, with the
disarming half subordinate. A reader meets the confound, forms the objection,
and answers — the clause that makes it harmless arrives after the objection has
already formed.

**Before**

> Rainfall in each plot's season ran a fifth higher than the year before, and
> every plot stood under cover, on a fixed watering schedule.

**After**

> Every plot stood under cover all season, watered only on a fixed schedule.
> Rainfall in each plot's season ran a fifth higher than the year before.

The reader is told the rain could not reach the plants before being told it
rained. Twenty such sentences, one per family.

**Applied to every item carrying the block clause, not only to `coherent_true`** —
and this is a deliberate departure from the brief. The block clause sits in
exactly two cells per family, one TRUE and one FALSE. Rewriting only
`coherent_true` would have made the construction itself a perfect predictor of
TRUE for those items: the lexical giveaway the build gates against. Rewriting all
four keeps it balanced, and the word-count shift lands equally on every cell for
the same reason.

### FALSE items a reader walked past (D-035)

Every one of the five near-invisible items at baseline was `broken_chronology` —
and the hunter audit had rated all five "found". A date inversion inside a case
line is exactly what a reader skims and a hunter, told to look, does not.

Two fixes, neither adding vocabulary:

**Round 1, adjacency.** The treatment date and the measurement date sat ten words
apart with the outcome between them. They are now adjacent, so the comparison is
one glance rather than a scan:

> before — fitted 4 May, ran 340 hours longer before failure **by 27 March** than…
> after — fitted 4 May, **measured by 27 March**, ran 340 hours longer before failure than…

**Round 2, a second cue.** A third case is now inverted as well: three of four
cases carry a measurement dated before the treatment, instead of two. An
inversion swaps two date tokens between positions rather than introducing any, so
it costs nothing lexically. The answer keys were updated to match.

No causal language was restored and nothing moved to the final sentence. Both
were off the table and both would have worked.

<!--MECHANISM-->
<!--/MECHANISM-->

### Duplicate flaw phrasing (D-033)

The brief asked that no two FALSE items share flaw phrasing. Scanning all 1225
pairs found 30 near-duplicates within families, 1 across, and zero shared
8-grams across families.

**Every one of the 30 within-family duplicates is a scope family's false trio** —
which *is* the matched-mechanism control. Section 2 of the analysis, the number
quoted when someone asks whether the coherent-false items are simply easier to
catch, holds by making the falsifying sentence identical across those cells so
that nothing but coherence differs. Breaking that identity would put phrasing
back into the one endpoint built to contain nothing but coherence.

So `no_duplicate_flaw_phrasing` enforces the rule **across** families and exempts
it **within** one, and says so in its own docstring. The single real cross-family
pair — two agricultural families that had both landed on rainfall as their stated
confound — was fixed. If the stricter reading is wanted anyway, the exemption is
one condition in the check and the gate is already written to fail without it.

## 5. Every gate

<!--GATES-->
<!--/GATES-->

### Two bugs the gates caught

**`src/flaws.py` read the wrong sentence.** It located the scope clause as the
*second* sentence of the mid-passage line. Adding the protection sentence makes
three, so it silently returned the confound sentence for every blocked item. It
now anchors on the last sentence, stable under both shapes.

**`src/synth.py` built a midline and never used it.** Every synthetic item went
out as six lines with no falsifying text in it at all, so every gate that reads
the mid-passage line had never been exercised end to end — the entire purpose of
the synthetic set. Fixed, and its clause pools now carry one entry per family
sharing a mechanism, so the set can pass the phrasing gate it exists to exercise.
The fixture in `tests/test_validate.py` had the same problem and the same fix.

## 6. For your co-founder

> **"How do you know a normal reader can find these flaws, and that the true
> items read as true?"**
>
> Until this week we didn't. We had a number — 94% of flaws found — but it came
> from an auditor we had *told* a flaw might be there and asked to name it. That
> measures how explicit a defect looks to someone already looking at it. It says
> nothing about whether anyone looks. When one of us read ten items cold, we
> caught one flaw in five. Against 94%, that has probability 6×10⁻⁵. The two
> numbers were never measuring the same thing.
>
> So there is now a second audit. It shows a reader **exactly** the prompt the
> model gets — same instruction, same claim, same passage, same three options —
> and nothing else. No hint that anything might be wrong, no box to describe a
> flaw in, because the model has no such box either. All we record is Yes, No or
> Unsure. Three independent passes per item. That gives a real per-item
> findability rate, and it is now the covariate the analysis conditions on;
> the old salience number stays in the report labelled as what it is.
>
> It immediately found two things inspection had missed. Every FALSE item a
> reader walked straight past was the same mechanism — a date inversion buried
> inside a case line — and the hunter had scored all of them "found". And our
> TRUE items had a false-positive rate of 0.22 in the coherent cells against
> 0.07 in the diverse ones, because we were stating a confound and then
> disarming it in a subordinate clause the reader never got to. Both are fixed:
> the protective fact now leads its own sentence, and the date inversion sits
> adjacent instead of ten words away, with a third case inverted as a second cue.
>
> Where that leaves us is in the tables above — measured, not asserted, with
> before-and-after on every cell and a gate that fails the build on regression.
> Two things I want to flag rather than bury. **The readers are language models,
> not people**; they are a far better proxy than a hunter but they are still a
> proxy, and the one human sample we have is n=10. **And the FALSE cells are not
> yet at parity** — the number and the gap are in section 3, and we stopped at
> the round limit rather than editing items until the number looked right.

## 7. Re-review set

`review/blind_items_2.md` — 10 items, one per family, from ten families that
appear in neither of the first two reviewed. Five TRUE, five FALSE, balanced
deliberately so the base rate teaches nothing. Claim, passage and question only;
cell, ground truth, mechanism, salience, reader rate and id all withheld, order
shuffled, and the generator refuses to write the file if any of those strings
appear in it.

The key is in `results/review_key/blind_items_2.json`. Answer first.

## 8. Commands

```bash
# rebuild items after any clause change (order matters)
.venv/Scripts/python.exe scripts/assemble_passages.py
.venv/Scripts/python.exe scripts/build_decorative.py
.venv/Scripts/python.exe scripts/apply_complexity.py

# the reader audit
.venv/Scripts/python.exe -m src.audit_reader make --run 0   # and 1, 2
#   ... run the readers over results/reader_audit/run*/blind/ ...
.venv/Scripts/python.exe -m src.audit_reader score
.venv/Scripts/python.exe scripts/apply_reader_rates.py

# gates and report
.venv/Scripts/python.exe -m src.validate --items items/draft items/seed
.venv/Scripts/python.exe scripts/lexical_ablation.py
.venv/Scripts/python.exe scripts/build_reader_report.py
.venv/Scripts/python.exe scripts/make_blind_review.py
```
