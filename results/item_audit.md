# Item audit — blind flaw identification

Step 9. Every item was shown to independent auditors carrying **only** the claim, the passage and the forced question — no cell label, no `ground_truth`, no `confound_note`, and never two items from the same family in one batch. The TRUE items were mixed in as decoys, so an auditor could not know that everything it saw was broken.

## Headline

| | count | of | rate |
|---|---|---|---|
| FALSE items whose intended flaw was found | 40 | 40 | 100% |
| **FALSE items where NO auditor found the flaw — BROKEN** | **0** | 40 | 0% |
| **FALSE items rated as good as stated outright — TOO EASY** | **16** | 40 | 40% |
| TRUE items an auditor invented a flaw for (false positives) | 0 | 40 | 0% |

**How to read the false-positive rate.** Auditors reported a flaw in 0 of 40 items that have nothing wrong with them. Finding a flaw in a FALSE item is only as informative as that rate is low; treat it as the audit's own noise floor, and discount the detection rate above accordingly.

**How to read the 'too easy' count.** The threshold is a mean explicitness of 4.0 out of 5, where 4 means "obvious; the passage all but says it". That bar is arguably unfair to `coherent_false` by construction: the design *requires* the shared confound to be stated in the passage, so an auditor will always see it once they look. Nothing scored 5 ("the passage states the problem outright"). The number to act on is not this count but the asymmetry below.

## The asymmetry that matters

| group | n | mean explicitness (1-5) |
|---|---|---|
| flaw_type `claim_mismatch` | 10 | 2.85 |
| flaw_type `shared_confound` | 20 | 3.90 |
| flaw_type `temporal` | 10 | 2.55 |
| cell `coherent_false` | 20 | **3.90** |
| cell `diverse_false` | 20 | **2.70** |

**The flaw in `coherent_false` is 1.20 points more salient than the flaw in `diverse_false`, on a 5-point scale.** That is not a wording accident. It is DECISIONS.md D-004 showing up in the data: a shared confound has to be *stated* in the passage for it to be a shared confound at all, while reversed dates and a substituted quantity are things a reader has to notice for themselves.

**What it does to the result.** AUC is computed within each coherence condition, so this predicts AUC(coherent) > AUC(diverse) from flaw salience alone, with no coherence effect involved. Note the direction: the hypothesis predicts that coherence *inflates* confidence and therefore *depresses* AUC(coherent). This artifact pushes the other way, so it makes the test **conservative** — a coherence effect found in spite of it is stronger evidence, not weaker. It still has to be reported.

**Two things already in place that separate them.** (1) `coherence_effect_within_true`, which `src/analyze.py` reports separately, contrasts two cells that contain no flaw at all, so flaw salience cannot touch it. (2) `diverse_false` is split evenly between `temporal` and `claim_mismatch` and each item is tagged, so the two can be compared against `coherent_false` separately rather than pooled.

## FLAGGED - broken items (auditors could only gesture, not identify)

_None._

## FLAGGED — broken items (no auditor found the intended flaw)

_None._

## FLAGGED — too easy (flaw rated as good as stated outright)

Explicitness >= 4 of 5 means auditors felt the passage announced the problem. These items measure reading rather than reasoning and will compress the truth effect. Soften the wording.

### `fam_bearing__coherent_false`  (shared_confound)

- code `7A9B21`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four machines are lathes at Bexley on the night shift, so one change to that plant's lubricant supply covers every case at once. The supply switched to a higher-grade synthetic mid-cycle, and the switch reaches every machine here because every machine drew from that supply. Better lubricant extends bearing life on its own, and each machine is compared only against itself in the previous cycle, before the switch, rather than against a machine running the same lubricant over the same period, so nothing here separates the redesign from the change of grade. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same switch but has every machine running on a sealed charge, so the new lubricant never reaches those bearings, which is what makes it sound.
- auditor said: A concurrent change is confounded with the bearing: the lubricant supply 'switched to a higher-grade synthetic mid-cycle, and every machine here drew from that supply', so all four machines (Bexley lathes, nights, fitted 4-7 March) got better lubrication at the same time as the new bearing; the extra 260-410 hours could be the lubricant.
- auditor said: A named alternative cause covers all four cases: the lubricant supply switched to a higher-grade synthetic mid-cycle and 'every machine here drew from that supply', so the extra 260-410 hours before failure could be the better lubricant rather than the redesigned bearing. The report's own claim that 'nothing else differed' contradicts the sentence immediately before it.

### `fam_coolant__coherent_false`  (shared_confound)

- code `5A0D9C`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four runs sit in the same workshop on the same night shift inside one quarter, so a single retooling covers every case at once. The shop refitted its machines with a faster insert grade that quarter, and the retooling reaches every case because each logged run was cut on one of those refitted machines. Each run is compared against itself last quarter, when the older grade was still fitted, so nothing in the log separates the coolant from the inserts. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same refit but cuts every logged run on a machine hired from outside, so the refit cannot reach the runs, which is what makes it sound.
- auditor said: The report states that every shop refitted its machines with a faster insert grade that same quarter and that every logged run was cut on one of those machines, so the faster inserts are an alternative cause fitting all four 4-7 minute reductions; the closing claim that 'no other change was made to any run' contradicts this.
- auditor said: A named alternative cause covers all four runs: "Every shop refitted its machines with a faster insert grade that quarter, and every logged run was cut on one of those machines." A faster insert grade directly shortens cycle time, so the 4-7 minute reductions cannot be attributed to the coolant.

### `fam_driptape__coherent_false`  (shared_confound)

- code `0AAF54`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four blocks sit on one farm in one summer, so a single rainfall record covers every case at once. Rain at roughly double the year before is one alternative cause, and it reaches every block because every block lay in open ground and took that rain as it fell. Each block is compared against itself a season earlier rather than against an undripped block in the same season, so nothing in the evidence separates the drip line from the rain. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same doubled rainfall but puts every block under a rain shelter fed only by metered water, so the rain cannot reach the crop and the season-over-season comparison holds.
- auditor said: Rainfall in each block's season 'ran to roughly double the year before' and 'every block lay in open ground, taking rain as it fell', so a doubling of water supply is an alternative cause that applies to all four blocks; the 0.4-0.7 t/ha gains cannot be attributed to the drip line.
- auditor said: Rainfall in each block's season ran to roughly double the year before, and the report says every block lay in open ground 'taking rain as it fell', so the doubled rainfall reached all four blocks and is an alternative cause of the 0.4-0.7 t/ha gains that fits every case.

### `fam_fertilizer__coherent_false`  (shared_confound)

- code `237197`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four plots sit in the same backyard in the same summer, so one weather history covers them all. Rainfall roughly double the year before is a single alternative cause, and it reaches every plot because every plot stood in the open and was watered by rain alone. Each plot is compared against itself a year earlier rather than against an untreated plot in the same year, so nothing in the evidence separates the fertilizer from the rain. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same record rainfall but puts the plots under cover on a fixed watering schedule, which is what makes it sound.
- auditor said: Rainfall in every plot's season was 'the highest on record, roughly double the year before' and every plot was 'watered by rain alone', so the doubled water supply is a live alternative cause that fits all four cases (8-13 cm gains); the comparison is against the previous, drier summer.
- auditor said: Rainfall in each plot's season was 'the highest on record, roughly double the year before', and the plots 'stood in the open, watered by rain alone' - so the doubled rainfall applied to every plot and fully explains the 8-13 cm height gain over the previous summer without any contribution from the fertilizer.

### `fam_filter__coherent_false`  (shared_confound)

- code `FE036E`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four kilns sit at Kestrel inside Q1, so one production history covers every case at once. Throughput fell by half against the earlier quarter, and the slowdown reaches every case because each stack fired in step with site output; burning half as much material lowers particulate on its own. Because each unit is compared against itself a quarter earlier rather than against an unupgraded unit in the same quarter, nothing in the log separates the filter upgrade from the slowdown. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same halved throughput but has every stack firing on a fixed duty cycle, so the slowdown cannot reach the readings, which is what makes it sound.
- auditor said: The report states throughput at each site 'fell by half against the earlier quarter' and that 'every stack logged here fired in step with site output', so the 8-14 percent lower particulate readings are fully explained by kilns firing half as much, not by the filter upgrade.
- auditor said: 'Throughput at each site fell by half against the earlier quarter, and every stack logged here fired in step with site output' - a 50 percent drop in firing fully explains particulate falling 8-14 percent without any filter effect.

### `fam_handwash__coherent_false`  (shared_confound)

- code `CC1BF2`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four entries sit in one hospital on one surgical floor inside the same first quarter, so a single contract-level change covers every case at once. Contract deep-cleaning hours ran at triple the earlier quarter, and that extra cleaning reaches every entry here because every ward was cleaned under the contract. Deep cleaning lowers ward infections on its own, and each ward is compared only against itself a quarter earlier, before the hours were tripled, rather than against a ward cleaned to the same standard in the same quarter, so nothing in the evidence separates the handwashing protocol from the extra cleaning. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same tripled contract hours but has every ward cleaned by in-house staff, so the contractor's change cannot reach these wards, which is what makes it sound.
- auditor said: A named alternative cause fits all four cases: contract deep-cleaning hours ran at triple the earlier quarter and 'every ward here was cleaned under that contract', so tripled deep-cleaning explains the 6-11 percent infection drop just as well as the handwashing protocol.
- auditor said: A named alternative cause covers all four cases: contract deep-cleaning hours ran at 'triple the earlier quarter' during exactly the period measured, and the report states 'every ward here was cleaned under that contract'. The 6-11 percent infection drop is equally attributable to tripled deep cleaning as to the handwashing protocol.

### `fam_inhaler__coherent_false`  (shared_confound)

- code `C6F63B`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four groups are the 12-17 band at Harrowfield inside one winter quarter, so a single district air history covers every case at once. Pollen and airborne irritants in that district falling by half from the previous quarter is one alternative cause of fewer attacks, and it reaches every group because every group lived and slept in unfiltered district air. Each band is compared against itself a quarter earlier rather than against an untreated band in the same quarter, so nothing in the evidence separates the inhaler from the drop in irritants. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) names the same halving of pollen and irritants but has every group living and sleeping on filtered indoor air, so the drop never reaches the patients and the quarter-over-quarter comparison holds.
- auditor said: Pollen and airborne irritants in each group's district 'fell by half from the previous quarter' and 'every group lived and slept in unfiltered district air', so an exposure drop of that size fully accounts for the 4-7 fewer attacks in all four groups without the inhaler doing anything.
- auditor said: A named alternative cause fits all four cases: 'Pollen and airborne irritants in each group's district fell by half from the previous quarter,' and every group 'lived and slept in unfiltered district air,' so the patients were fully exposed to that halving. A 50% drop in the asthma trigger explains the 4-7 fewer attacks without the inhaler doing anything.

### `fam_labkit__coherent_false`  (shared_confound)

- code `180003`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four sections sit at Harlow in one autumn term, so a single change to the central syllabus covers every case at once: coursework's share of the final grade doubled against the previous year. That change reaches all four sections because every one of them was marked to that central syllabus. Each section is compared against itself a year earlier under the old share, so the kit and the doubled share moved together and the 9, 6, 11 and 7 point rises cannot be attributed to the kit. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same doubling but marks every section to a frozen departmental syllabus, so the change cannot reach them, which is what makes it sound.
- auditor said: The central syllabus 'doubled coursework's share of the final grade against the previous year' and every section was marked to that syllabus, so a doubled grade incentive applies to all four sections and explains a 6-11 point rise in coursework completion without the lab kit.
- auditor said: The central syllabus 'doubled coursework's share of the final grade against the previous year' and 'every section was marked to that central syllabus', so all four sections were exposed to a grading change that by itself would raise coursework completion; it fits all four cases and is not separable from the lab kit.

### `fam_onboarding__coherent_false`  (shared_confound)

- code `338FC3`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four cohorts are starter signups in Norway inside the same March window, so one trial policy covers every case at once. The free trial ran thirty days, four times its length in the earlier period, and a thirty-day trial leaves users still inside their trial at week two, which raises week-two retention on its own. That change reaches all four cohorts because every cohort here signed up on that trial. Each cohort is compared against itself a month earlier, when the shorter trial still applied, so nothing in the evidence separates the email from the trial length. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) names the same thirty-day trial but has every cohort sign up on a paid plan, so the trial length cannot reach them and the comparison holds.
- auditor said: The free trial was lengthened to thirty days, 'four times its length in the earlier period, and every cohort here signed up on that trial', so the same period that introduced the email also quadrupled trial length for all four cohorts; a longer trial plainly raises week-two retention on its own.
- auditor said: A named alternative cause fits all four cohorts: the free trial was lengthened to thirty days, 'four times its length in the earlier period', and 'every cohort here signed up on that trial'. A trial four times longer would by itself lift week-two retention, so the 3-6 point gain cannot be attributed to the onboarding email.

### `fam_pricing__coherent_false`  (shared_confound)

- code `D14DA7`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four lines are coffee at the Harlow Street store inside the same month, so one shelf covers every case at once. Premium stock's share of that shelf hit a record, double the earlier period, and a dearer average item lifts revenue per order on its own. That shift reaches all four lines because every line was stocked straight from the shelf as it stood. Each line is compared against itself a month earlier, before the mix moved, so nothing here separates the new pricing rule from what was on the shelf. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) names the same record premium share but stocks every line from a fixed year-round range, so the shift cannot reach the lines and the comparison holds.
- auditor said: Premium stock's shelf share doubled to a record over the same period and 'every line was stocked straight from that shelf', so a richer premium product mix explains the 4-7 percent rise in revenue per order for all four lines independently of the pricing rule.
- auditor said: A named alternative cause covers all four lines: "Premium stock's share of each store's shelf hit a record, double the earlier period, and every line was stocked straight from that shelf." Doubling premium mix raises revenue per order on its own, so the 4-7 percent lift is not attributable to the pricing rule.

### `fam_routing__coherent_false`  (shared_confound)

- code `ED27C1`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four groups run out of Kettleby Wharf in February with the same vans and the same dispatcher, so one order book covers every case at once. Volume through that depot hit a record low, half the earlier period, and a lighter book means fewer drops on each run, which cuts the time a run takes on its own. That drop reaches all four groups because every run carried whatever drop count the depot set. Each group is compared against itself a month earlier, when the book was full, so nothing in the evidence separates the routing change from the thinner load. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) names the same record low volume but holds every run to the same drop count as before, so the volume drop cannot reach the runs and the comparison holds.
- auditor said: Order volume 'hit a record low, half the earlier period' and 'every run carried whatever drop count the depot set', so the runs were exposed to the halved workload; fewer drops per run explains the 5-9 minute reduction independently of the routing change.
- auditor said: Order volume through each depot 'hit a record low, half the earlier period', and 'every run carried whatever drop count the depot set' - so each run carried roughly half the stops it did a month earlier, which alone explains delivery times running 5-9 minutes below the earlier period regardless of the routing change.

### `fam_seedcoat__coherent_false`  (shared_confound)

- code `A9D2D3`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four lots sit at one nursery in one spring, so a single house shares its air with every case at once. A house running a full four degrees warmer than the earlier sowing is one alternative cause, and warmth is a direct driver of germination; it reaches every lot because every lot sat on an open bench taking that house air directly. Each lot is compared against the same lot a season earlier rather than against an uncoated lot on the same warm bench, so nothing in the evidence separates the coating from the heat. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same four-degree rise but puts every lot in a sealed cabinet held at a set temperature, so the warmer air cannot reach the seed and the season-over-season comparison holds.
- auditor said: The report says 'House air temperature ran a full four degrees above the earlier sowing, and every lot sat on an open bench, taking the house air directly', so a four-degree warmer glasshouse is an alternative cause of the 6-11 point germination gain that applies to all four lots.
- auditor said: 'House air temperature ran a full four degrees above the earlier sowing, and every lot sat on an open bench, taking the house air directly' - the warmer house, not the coating, plausibly accounts for the 6-11 point germination gain in all four lots.

### `fam_sleepapp__coherent_false`  (shared_confound)

- code `4969DF`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four cohorts are 40 Fenwick users on Corvid 7 phones inside one March, so a single weather history covers every case at once. Outdoor temperature in that city falling ten degrees from the month before is one alternative cause of a lower resting heart rate, and it reaches every cohort because every cohort slept in unconditioned rooms open to outside air. Each cohort is compared against itself in February, when the nights were warmer, rather than against an untracked cohort in the same month, so the cooling and the app arrived together and the evidence cannot separate them. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) names the same ten-degree fall but has every cohort sleeping in rooms held at a fixed 20 degrees, so the outdoor change never reaches them and the month-over-month comparison holds.
- auditor said: A named alternative cause covers all four cohorts: outdoor temperature in each cohort's city fell ten degrees from the month before and 'every cohort slept in unconditioned rooms open to outside air', so cooler sleeping conditions could produce the 3-6 bpm drop in resting heart rate independently of the app.
- auditor said: A named alternative cause fits all four cohorts: 'Outdoor temperature in each cohort's city fell ten degrees from the month before, and every cohort slept in unconditioned rooms open to outside air.' A ten-degree drop in sleeping-room temperature is itself a known depressor of resting heart rate, so the 3-6 bpm fall is confounded with the cooler rooms.

### `fam_solder__coherent_false`  (shared_confound)

- code `0899C1`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four runs sit on the Aspen line inside March, so they share one inspection regime and a single change to it covers every case at once. The optical inspection threshold was loosened two grades against the earlier period, and that looser setting reaches every run here because every run was graded at that optical stage. A looser threshold records fewer defects on identical boards, and each run is compared against the same line in February, when the tighter setting was still in force, so the drops of 6, 4, 7 and 5 percent follow from the setting as readily as from the solder profile and nothing here separates them. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same loosening but has every run graded at electrical test, so the optical setting cannot touch those counts, which is what makes it sound.
- auditor said: 'The optical inspection threshold was loosened two grades against the earlier period, and every run here was graded at that optical stage', so the 4-7 percent drop in measured defect rate is explained by the looser detection criterion rather than by fewer actual defects from the solder profile.
- auditor said: The measuring instrument changed at the same time as the profile: 'The optical inspection threshold was loosened two grades against the earlier period, and every run here was graded at that optical stage.' A looser threshold mechanically records fewer defects, which accounts for the 4-7 percent drop without any real change in board quality.

### `fam_timetable__coherent_false`  (shared_confound)

- code `FAE756`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four cases are one year group at one school inside a single autumn term, so they share one district and one attendance reward scheme. That scheme tripled its payment over the term, a single alternative cause, and it reaches every case because all four cohorts were on the district roll the scheme covered. Because each cohort is compared only against itself a term earlier, when the smaller payment applied, nothing in the register separates the reward from the timetable change. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same tripled payment but puts every cohort on an independent roll the scheme never covered, so the reward cannot reach them, which is what makes it sound.
- auditor said: The district 'attendance reward scheme ... tripled its payment that term' and 'every cohort was on the district roll that scheme covered', so a tripled cash incentive for attending covers all four cohorts and accounts for the 4-7 percent absence drop independently of the timetable.
- auditor said: The district 'attendance reward scheme ... tripled its payment that term' and 'every cohort was on the district roll that scheme covered', so all four cohorts got a large new financial incentive to attend during exactly the measurement window; that alone explains the 4-7 percent absence drop.

### `fam_tutoring__coherent_false`  (shared_confound)

- code `BCC78C`, 2 verdicts, answers ['No', 'No'], matches ['same_flaw', 'same_flaw']
- mean explicitness 4.0 / 5
- **intended flaw:** All four cohorts sit at one college, in one subject, in one spring term, so a single change to the board's pass mark covers every case at once. The mark fell a full grade below the previous term's, and it reaches all four cohorts because every one of them sat the board's own paper under that same mark. Each cohort is compared against itself in the previous term, when the higher mark applied, so the easier standard and the tutoring blocks moved together and the 4 to 8 point gains cannot be attributed to the blocks. The TRUE item in this family carrying closer_variant 'changed_block' (coherent_true in half the families and diverse_true in the other half, see D-022) states the same drop in the pass mark but has every cohort sit an in-house paper under a fixed internal mark, so the drop cannot reach them, which is what makes it sound.
- auditor said: 'The board's pass mark fell a full grade below the previous term's, and every cohort sat the board's own paper under that same mark' - the pass threshold was lowered for exactly these four cohorts, which raises pass rates by 4-8 points regardless of tutoring, and pass rate is not a measure of learning here.
- auditor said: The report says 'the board's pass mark fell a full grade below the previous term's, and every cohort sat the board's own paper under that same mark' - an easier pass threshold applied to all four cohorts and by itself raises pass rates, so the 7, 5, 8 and 4 point gains cannot be attributed to the tutoring blocks.

## Worth a look — an auditor named a DIFFERENT flaw

_None._

## TRUE items an auditor thought were broken

_None._

## Detection by flaw type

| flaw_type | n | found | missed | too easy |
|---|---|---|---|---|
| claim_mismatch | 10 | 10 | 0 | 0 |
| shared_confound | 20 | 20 | 0 | 16 |
| temporal | 10 | 10 | 0 | 0 |

## Every item

| item | cell | flaw | bucket | answers | explicitness |
|---|---|---|---|---|---|
| `fam_bearing__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_bearing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_bearing__diverse_false` | diverse_false | temporal | found | No,No | 2.0 |
| `fam_bearing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_checkout__coherent_false` | coherent_false | shared_confound | found | No,No | 3.5 |
| `fam_checkout__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_checkout__diverse_false` | diverse_false | claim_mismatch | found | No,No | 3.0 |
| `fam_checkout__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_coolant__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_coolant__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_coolant__diverse_false` | diverse_false | temporal | found | No,No | 3.0 |
| `fam_coolant__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_driptape__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_driptape__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_driptape__diverse_false` | diverse_false | temporal | found | No,No | 2.0 |
| `fam_driptape__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_fertilizer__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_fertilizer__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_fertilizer__diverse_false` | diverse_false | temporal | found | No,No | 2.0 |
| `fam_fertilizer__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_filter__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_filter__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_filter__diverse_false` | diverse_false | claim_mismatch | found | No,No | 3.0 |
| `fam_filter__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_handwash__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_handwash__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_handwash__diverse_false` | diverse_false | claim_mismatch | found | No,No | 2.0 |
| `fam_handwash__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_inhaler__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_inhaler__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_inhaler__diverse_false` | diverse_false | temporal | found | No,No | 3.0 |
| `fam_inhaler__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_labkit__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_labkit__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_labkit__diverse_false` | diverse_false | temporal | found | No,No | 3.0 |
| `fam_labkit__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_nestbox__coherent_false` | coherent_false | shared_confound | found | No,No | 3.5 |
| `fam_nestbox__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_nestbox__diverse_false` | diverse_false | temporal | found | No,No | 2.5 |
| `fam_nestbox__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_onboarding__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_onboarding__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_onboarding__diverse_false` | diverse_false | temporal | found | No,No | 3.0 |
| `fam_onboarding__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_physio__coherent_false` | coherent_false | shared_confound | found | No,No | 3.5 |
| `fam_physio__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_physio__diverse_false` | diverse_false | claim_mismatch | found | No,No | 3.0 |
| `fam_physio__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_pricing__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_pricing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_pricing__diverse_false` | diverse_false | claim_mismatch | found | No,No | 3.0 |
| `fam_pricing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_reading__coherent_false` | coherent_false | shared_confound | found | No,No | 3.5 |
| `fam_reading__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_reading__diverse_false` | diverse_false | temporal | found | No,No | 2.0 |
| `fam_reading__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_routing__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_routing__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_routing__diverse_false` | diverse_false | claim_mismatch | found | No,No | 3.0 |
| `fam_routing__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_seedcoat__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_seedcoat__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_seedcoat__diverse_false` | diverse_false | claim_mismatch | found | No,No | 2.5 |
| `fam_seedcoat__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_sleepapp__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_sleepapp__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_sleepapp__diverse_false` | diverse_false | temporal | found | No,No | 3.0 |
| `fam_sleepapp__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_solder__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_solder__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_solder__diverse_false` | diverse_false | claim_mismatch | found | No,No | 2.5 |
| `fam_solder__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_timetable__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_timetable__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_timetable__diverse_false` | diverse_false | claim_mismatch | found | No,No | 3.0 |
| `fam_timetable__diverse_true` | diverse_true |  | clean | Yes,Yes |  |
| `fam_tutoring__coherent_false` | coherent_false | shared_confound | too_easy | No,No | 4.0 |
| `fam_tutoring__coherent_true` | coherent_true |  | clean | Yes,Yes |  |
| `fam_tutoring__diverse_false` | diverse_false | claim_mismatch | found | No,No | 3.5 |
| `fam_tutoring__diverse_true` | diverse_true |  | clean | Yes,Yes |  |

## Limitation, stated plainly

The auditors and the item authors are the same model family (DECISIONS.md D-017). This measures whether the flaws are **findable**, not how hard they are for the model under test — an author and an auditor that share priors will agree more than two independent readers would. Re-running the audit with a different model, or handing the drafts to a person, is the way to get a number that means more than this one.
