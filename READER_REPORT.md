# READER REPORT — measuring findability instead of assuming it

## Provenance

The three blind reader passes in `results/reader_audit/` were answered by
language-model readers — subagent instances of the coding agent that built
this repo; the audit artifacts do not record a specific model name (D-032,
D-037) — shown only the scored prompt (third option `Unsure` at audit time;
renamed `Unknown` on Kaggle for tokenization, D-019). The 1-5 salience
ratings were made by the item author.

---

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
| cell | n | hunter per-auditor | reader catch (baseline) | reader catch (now) | 95% CI, clustered by reader |
|---|---|---|---|---|---|
| `coherent_false` | 20 | 39/40 = 98% | 1.00 | **0.97** | [0.74, 0.99] |
| `diverse_false` | 20 | 36/40 = 90% | 0.78 | **0.90** | [0.82, 1.00] |
| `decorative_false` | 10 | 19/20 = 95% | 1.00 | **1.00** | [1.00, 1.00] |

| cell | n | reader false-positive (baseline) | now | 95% CI, clustered by reader | readers giving 0 |
|---|---|---|---|---|---|
| `coherent_true` | 20 | 0.22 | **0.25** | [0.06, 0.41] | 13/17 |
| `diverse_true` | 20 | 0.07 | **0.03** | [0.00, 0.04] | 22/24 |
| `decorative_true` | 10 | 0.10 | **0.03** | [0.00, 0.14] | 20/21 |
<!--/HUNTER_VS_READER-->

Read the **hunter per-auditor** column against **reader catch (baseline)**. That
gap is the whole point of this pass: the hunter was near ceiling on items a plain
reader was walking past. The **reader catch (now)** column is what the rewrites
in section 4 bought — the two columns have converged, which is the result rather
than the premise.

The hunter numbers here were re-measured on the *current* items, not carried
over. The figures on disk described passages that fifty items no longer have,
and putting a stale hunter beside a fresh reader would have measured method and
content at the same time. A hunter report counts as a find only if it **quoted
the falsifying span** — a shared 5-gram with `flaws.flaw_sentence` — rather than
merely asserting something was wrong. That deterministic rule reproduces the
earlier judged figure exactly at 94/100, and reported and matched are identical,
so every flaw the hunter reported was the right one.

### The intervals are clustered by reader, and that matters

One reader agent answers every item in a batch, so **a batch is a cluster and
verdicts inside it are not independent**. Treating 300 reads as 300 observations
would give an interval several times too narrow. The CI column resamples
clusters.

This is not a technicality. Between two rounds, on items where exactly one
`coherent_true` passage had changed, that cell went from **0.00 to 0.25** — and
the 15 dissenting reads were perfectly concentrated in 4 of 18 batches, at 4/4,
4/4, 3/5 and 4/4, with zero everywhere else. A handful of readers reject the
whole cell; the rest reject none of it. `n = 300 reads` is really `n = 18
readers`, and any single round's cell mean is a draw from a bimodal process.

The strict readers are **not** simply strict: in the same batches they rejected
`coherent_true` at 1.00, 1.00, 0.60 and 1.00 while rejecting `diverse_true` at
0.12, 0.00, 0.00 and 0.00. Something specific to `coherent_true` is what flips.
Section 4 says what it is.

## 3. Triage, before and after

<!--TRIAGE-->
| cell | metric | before | after | invisible / reads-as-false |
|---|---|---|---|---|
| `coherent_true` | false-positive rate | 0.22 | **0.25** | 0 -> 3 |
| `coherent_false` | catch rate | 1.00 | **0.97** | 0 -> 0 |
| `diverse_true` | false-positive rate | 0.07 | **0.03** | 1 -> 1 |
| `diverse_false` | catch rate | 0.78 | **0.90** | 1 -> 0 |
| `decorative_true` | false-positive rate | 0.10 | **0.03** | 0 -> 0 |
| `decorative_false` | catch rate | 1.00 | **1.00** | 0 -> 0 |

| target | before | after | met |
|---|---|---|---|
| every TRUE cell false-positive <= 0.15 | 0.22 | **0.25** | **NO** |
| coherent_true - diverse_true gap <= 0.10 | 0.15 | **0.22** | **NO** |
| every FALSE item catch >= 0.5 | 0.00 | **0.67** | YES |
| FALSE cell means within 0.10 | 0.22 | **0.10** | YES |
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
| mechanism | n | catch before | catch after |
|---|---|---|---|
| `broken_chronology` | 10 | 0.57 | **0.80** |
| `scope_mismatch` | 30 | 1.00 | **0.98** |
| `stated_confound` | 10 | 1.00 | **1.00** |
<!--/MECHANISM-->

### What is still not fixed, and why (D-036)

**The FALSE-item targets are met. The TRUE-item targets are not.** The triage
table says so. This section says why, because the reason is structural rather
than a wording miss.

The hunter's descriptions name the objection precisely, and it is the same one
every time, in every TRUE cell:

> "The scored population did not all receive the program: *Scores cover every
> pupil on each class roll, and a quarter of each roll missed at least one
> program session.*"

> "The outcome is measured on a population that largely did not receive the
> treatment: *Retention figures cover every signup, email opened or not, and a
> third of each cohort's signups never opened it.*"

That is the `whole_incomplete` clause pair — measure everyone, including
partial compliers. The design labels it TRUE, and on the merits that is
defensible: including non-compliers dilutes toward the null, so a positive
result measured that way is *conservative*, not inflated. Readers do not read it
that way. They read a quarter of the denominator not getting the treatment as a
defect, and say No.

**It cannot be softened, and this was tested rather than assumed.** The obvious
fix is to make the shortfall minor — one session rather than a quarter of them.
In the ten confound families `whole_incomplete` is carried only by TRUE items,
so softening there looked safe. It was applied to all ten and the build failed:

> `'single'` is 8T/0F across 8 families — a 100% split. A classifier trained on
> other families can carry that straight across a fold boundary.

Which is the gate being right. The information "the shortfall here is minor"
appears *only* in TRUE items, so any wording that conveys it is a giveaway,
whatever words it uses. In the ten scope families it is worse: the same clause
string carries the `subset_incomplete` flaw in 30 FALSE items, so softening it
would blunt the flaw the FALSE cells depend on. The experiment was reverted;
`results/scope_clauses.json` is back to its round-3 state and the passages are
byte-identical to the commit before it.

So this is a real tension in the balanced-clause design (D-026), not a typo:
**the same clause has to serve as innocuous in a TRUE item and as falsifying in a
FALSE one, and readers do not split it where the design does.** Three ways out,
none of them a wording change, all of them the reviewer's call:

1. Accept it and report `coherent_true` false-positive rate as a known
   property, carrying it into the analysis as the covariate it already is.
2. Drop `whole_incomplete` from the TRUE cells and rebalance the clause
   assignment from scratch — D-026 showed the current solution is the unique
   balanced one, so this means changing what the four cells are.
3. Re-label. If a careful reader reliably says a quarter non-compliance means
   the evidence does not establish the claim, the ground-truth label is the
   thing that is wrong.

Editing wording further would be chasing a number across a measurement whose
own interval is [0.06, 0.41]. Three of five rounds were used; the fourth was
spent on the experiment above, and it argued for stopping.

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
| gate | before (`8315d9d`) | after | measured now |
|---|---|---|---|
| `word_count_parity` | PASS | PASS | grand mean 185.3 words; largest cell deviation +7.66% (limit +/-10%) |
| `coherent_one_value_per_dimension` | PASS | PASS | 60 coherent + decorative items checked across their condition dimensions |
| `diverse_four_values_per_dimension` | PASS | PASS | 40 diverse items checked across their dimensions |
| `no_lexical_giveaway` | PASS | PASS | grouped 5-fold CV accuracy 49.3% (mean of 5 shufflings; max 51.1%; 0/5 over limit) (limit 60%, chance 50%) |
| `cell_balance` | PASS | PASS | 100 items, 20 families (10 with a control arm), per-cell {'coherent_false': 20, 'coherent_true': 20, 'diverse_false': 20, 'diverse_true': 20, 'decorative_false': 10, 'decorative_true': 10} |
| `no_duplicate_passages` | PASS | PASS | 100 distinct passages across 100 items |
| `no_answer_key_leakage` | PASS | PASS | 100 passages scanned for answer-key leakage |
| `passage_word_balance` | PASS | PASS | 1015 distinct words; flagged when \|T-F\| > 4, the split exceeds 40%, and the word spans >= 3 families. 0 flagged; worst qualifying split 17% |
| `flaw_declarations_complete` | PASS | PASS | 50 FALSE items checked |
| `matched_mechanism_subset` | PASS | PASS | scope_mismatch: 10 coherent_false, 10 diverse_false, over 10 shared families |
| `no_duplicate_flaw_phrasing` | — *(new)* | PASS | 50 FALSE items, 1185 cross-family pairs; worst similarity 0.62 (limit 0.7), no shared 8-grams; 40 within-family pairs exempt (matched mechanism, D-024) |
| `control_surface_match` | PASS | PASS | 20 decorative items; largest surface gap vs diverse +3.3% (limit +/-10%); condition values 4.0 vs coherent 4.0, diverse 16.0 |
| `all_items_unreviewed` | PASS | PASS | 100/100 items are 'unreviewed' |

**Component ablation** — grouped 5-fold CV accuracy at telling TRUE from FALSE on each part of the passage. Chance is 50%.

| passage part | accuracy |
|---|---|
| `full` | 49.9% |
| `lead` | 50.0% |
| `cases` | 50.8% |
| `closer` | 50.0% |
| `no_closer` | 49.8% |
| `no_dates` | 46.6% |
| `no_dates_no_numbers` | 46.6% |
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
> 0.07 in the diverse ones.
>
> **On the FALSE items, that is fixed and the answer to your question is yes.** A
> plain reader now catches 0.97, 0.90 and 1.00 across the three false cells,
> against a hunter that gets 0.94 — the reader has essentially caught up with
> the auditor who was told where to look. Every individual item is above 0.5 and
> the three cells sit within 0.10 of each other, which was the target.
>
> **On the TRUE items the answer is: not yet, and I can tell you exactly why.**
> `coherent_true` sits at 0.25 against a target of 0.15. The cause is not the
> coherence — it is one clause. Several of our TRUE items measure an outcome
> over everyone, including a fraction who did not fully take the treatment. That
> is deliberate and it is conservative, because including non-compliers biases
> toward finding nothing. Readers do not buy it; they see a quarter of the
> denominator untreated and call the evidence broken. We tried the obvious fix,
> making the shortfall small, and our own lexical gate rejected it: the phrase
> would then appear only in TRUE items and become a giveaway a classifier could
> learn. So it is a real design tension, not a typo, and section 4 lays out the
> three ways out. It is your call which.
>
> Two more things I want to flag rather than bury. **The readers are language
> models, not people** — a far better proxy than a hunter, but a proxy, and the
> one human sample we have is n=10 per review. **And the measurement is noisier
> than it looks**: one reader answers a whole batch, so ~300 reads is really a
> few dozen readers.
> Between rounds that cell went 0.00 to 0.25 with one item changed, because four
> readers reject the cell wholesale and the other fourteen reject none of it. We
> report the interval clustered by reader — [0.06, 0.41] — rather than the
> flattering round. We stopped editing at that point instead of tuning items
> until a noisy number came out under the line.

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

# after editing a few items: re-measure ONLY those, not all 100 (D-039)
.venv/Scripts/python.exe scripts/reader_audit_patch.py make
#   ... run the readers (scripts/wf_reader_patch.js) ...
.venv/Scripts/python.exe scripts/reader_audit_patch.py score
.venv/Scripts/python.exe scripts/apply_reader_rates.py

# gates and report
.venv/Scripts/python.exe -m src.validate --items items/draft items/seed
.venv/Scripts/python.exe scripts/lexical_ablation.py
.venv/Scripts/python.exe scripts/build_reader_report.py
.venv/Scripts/python.exe scripts/make_blind_review.py
```
