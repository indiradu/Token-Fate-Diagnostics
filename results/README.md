# Reported Results Manifest

This repository includes compact, inspectable aggregate artifacts behind the
reported tables and intervals. It intentionally excludes raw generation traces,
candidate rows, hidden-state arrays, locally fitted `.joblib` models, and
scheduler outputs; those files are unnecessary to inspect the reported metrics
and can contain complete model generations.

## Main Result Artifacts

| Directory | Contents | Role in the study |
| --- | --- | --- |
| `commitment_signal_tables_workshop/` | cross-task signal tables, baseline comparison, oracle summary | Broad trace-regret prediction across tasks and signals. |
| `robustness_gsm8k_frozen_selector_analysis/` | example-level fixed-selector bootstrap | Main positive high-confidence GSM8K result. |
| `robustness_gsm8k_highconf_analysis_final/` | residual-selector bootstrap and confidence-conditioned tables | Shows the residual-only selector is statistically tied with confidence. |
| `robustness_llada_gsm8k_general_paired_500/` | aggregate intervention summary and pair bootstrap | Strict GSM8K intervention for the frozen general selector. |
| `robustness_llada_gsm8k_highconf_paired_500/` | aggregate intervention summary and pair bootstrap | Strict GSM8K intervention for the residual selector. |
| `robustness_llada_math500_highconf_paired/` | disjoint MATH500 aggregate intervention and bootstrap artifacts | Cross-task exploratory replication. |
| `logit_lens_suite_30_r3/` | layer and slice summaries plus attention-probe results | Mechanistic logit-lens and attention diagnostics. |
| `token_fate_layerwise_wide_30_r2/probes/` | aggregate layerwise probe metrics and fate summaries | Hidden-state probe comparison. |

## PhaseLock Premise Audit (2026-08-31 / 09-01)

Counterfactual audit of the PhaseLock gates. Every run's
`committed_token_drift.csv` carries the premise-A evidence (post-commit drift
under a fixed identity). `clamp` runs additionally freeze a committed token's
hidden row against a drift-matched same-trajectory control (premises B and C);
`commit` runs additionally force a stable token to lock early against a
same-position late-commit control (the auxiliary timing test). All runs exclude
EOS/EOT candidates and use disjoint example offsets.

**Label note (realigned 2026-09-01, clarified 2026-09-05).** `A` is
*token-identity stability does not imply reference-state settlement* -- its
evidence is the post-commit drift in `committed_token_drift.csv` (identity is
fixed by construction because a transferred token is never remasked), not an
intervention. The meaningful version is drift beyond numerical/background
variation in the cached state active tokens would see, not merely nonzero hidden
movement. The `commit`-arm runs below measure the **auxiliary**
commitment-timing question -- is committing *earlier than the decoder would*
harmful? -- which is reported beside the ladder, not as premise A. `B` is an
existence claim about the set `C=1, R=0`: identity commitment is harmless while
reference freezing changes the output. Clamp candidates are drift-stratified by
construction, so no directory here contains an unbiased population base rate.
`C` should be read as "reference-state drift predicts freeze harm," with
attention exposure as the natural sensitivity proxy. `U` is a separate systems
proposition about latency and is **not tested** by any of these runs.

| Directory | Contents | Role in the study |
| --- | --- | --- |
| `settlement_fate_a_premise_math500_40x3_20260831/` | commit-arm summary, run config | Auxiliary commitment-timing test, mixed MATH500, gen 64, 120 pairs. Also carries premise-A drift rows. |
| `settlement_fate_a_premise_math500_100x3_20260831/` | commit-arm summary, run config | Auxiliary timing replication, 296 pairs; paired deltas are zero. Premise-A drift: 92.0% of 6,400 rows drift > 0.25. |
| `settlement_fate_a_premise_math500l5_40x3_g256_20260831/` | commit-arm summary, run config | Auxiliary timing test, level-5 gen 256; null, but candidate confidence is higher by construction. Premise-A drift: 96.7% of 10,240 rows drift > 0.25. |
| `settlement_fate_c_premise_math500_100x3_20260831/` | clamp-arm summary, run config | Premises B and C, mixed MATH500, gen 64, 300 pairs. |
| `settlement_fate_c_premise_math500l5_40x2_g256_20260831/` | clamp-arm summary, run config | Premises B and C, level-5 gen 256, 80 pairs; the run where C resolves. |
| `settlement_pairs_a_pooled_20260831/` | example-clustered bootstrap and sign tests | Auxiliary earliness attribution: `+0.7` pp non-target change, 95% CI `[0.0, +1.7]`; raw early-commit harm is `0.93%` (5 of 536). Not premise A. |
| `settlement_pairs_c100x3_20260831/` | example-clustered bootstrap and sign tests | Premise C at gen 64: `+3.3` pp, 95% CI `[-2.0, +8.7]`, unresolved. |
| `settlement_pairs_c_l5_g256_20260831/` | example-clustered bootstrap and sign tests | Premise C at level-5 gen 256: `+25.0` pp, 95% CI `[+12.5, +37.5]`, sign test `p = 0.0002`. |
| `settlement_pairs_a_l5_g256_20260831/` | example-clustered bootstrap | Auxiliary timing test at level-5 gen 256; all deltas exactly zero. |
| `compute_utility_proxy_c100x3_20260831/` | drift-gate policy sweep, harm-by-drift bins | Algorithmic proxy for the U proposition (row updates avoided, **not latency**): freeze-at-commit ceiling `20.8%`; `tau=0.3` keeps `13.7%`. U itself is untested. |
| `compute_utility_proxy_a100x3_20260831/` | drift-gate policy sweep, settlement timing | Independent replication of the sweep plus token-vs-reference settlement times. |

Every strict-intervention directory includes its aggregate group summary and
paired bootstrap comparisons where applicable.

## Prior-Method Lock Admission Audit (2026-09-01)

Whether published compute-lock methods (SureLock, LESS, TraceLock, TACG) are
entitled to assume reference-state settlement. They lock an *already unmasked*
position once its posterior stabilizes, so their lock step is later than
LLaDA's transfer step and premise A as measured above does not bind them. See
`docs/lock_admission_audit.md`.

| Directory | Contents | Role in the study |
| --- | --- | --- |
| `lock_admission_bound_phase0_20260901/` | predicate-free oracle bound vs savings budget | Phase 0, no new decode. At lock-at-commit the bound reproduces premise A (median `0.438`, 95% CI `[0.432, 0.444]`, 92.8% > 0.25 over 11,309 tokens). It decays to `0.062` once half the horizon is left unfrozen, so **the free bound cannot resolve delayed locks** and Phase 1 is required. Convicts only; never acquits. |
| `lock_predicate_audit_g64_20260901/` | threshold sweep, train-frozen thresholds, per-rule bounds | Phase 1 at gen 64. Thresholds frozen on offset 0 (n=100), reported on offset 407 (n=90); max train→eval savings drift `0.035`. |
| `lock_predicate_audit_l5g256_20260901/` | threshold sweep, self-split thresholds, per-rule bounds | Phase 1 at level-5 gen 256 (n=54), the regime where drift effects are visible. Thresholds fit on a disjoint 40% of the same run's examples. |

### Phase 1 findings at gen 64

1. **The admission rules do carry information beyond confidence.** At matched
   savings, SureLock-style KL and LESS-style JSD+persistence roughly halve the
   share of locked tokens with *provable* post-lock movement `> 0.25` relative
   to a confidence-only gate: at savings `~0.24`, `7.2%` vs `16.5%`; at savings
   `~0.55`, `15.3%` vs `30.6%`.
2. **Almost all of premise A's drift happens in the very first step after
   commit.** On this same eval run, max drift from the commit-step reference
   (premise A) is median `0.442` with `91.6%` above `0.25` — reproducing Phase 0
   on independent examples. Delaying the lock by a **single step** drops that to
   `0.266` and `55.0%`. This is why evaluating prior methods at LLaDA's transfer
   step would be a strawman, and it is quantified here rather than assumed.
3. **What the frontier costs is the finding.** As `tau` loosens these rules
   interpolate toward the earliest possible lock by construction — at savings
   `0.950` SureLock-style gives `0.254` / `51.3%`, essentially the sweep's floor
   of `0.266` / `55.0%`. That convergence is structural, not a defect anyone
   missed. The content is the **exchange rate**: cutting the provably-moving
   share from `51.3%` to `15.3%` costs about 40 points of savings (`0.950` to
   `0.548`). A drift-safety gate is expensive in exactly the currency the method
   exists to earn, which makes this a `U`-shaped question more than an
   `A`-shaped one. At the most conservative point (savings `0.093`), `4.5%` of
   admitted tokens still provably move `> 0.25`.
4. **No rule can be acquitted from this data.** The triangle-inequality upper
   bound is `0.55`–`2.22` at every operating point, so settlement is never
   certified anywhere; only the convicting lower bound speaks.

Two incidental notes. SureLock-style KL and LESS-style JSD+persistence behave as
near-duplicates here (both are adjacent-step divergences; the persistence
requirement adds little), and SureLock's optional confidence gate at `0.9` is
non-binding because post-commit confidence is almost always above it.

**Reading the rows.** No rule in the sweep can fire at the commit step, because
post-commit posteriors are logged from `base_step + 1` onward. The sweep's floor
row is therefore named `earliest_logged_lock`, not lock-at-commit, and
`summary.md` carries a separate `premise_A_lock_at_commit` reference row so the
two are not conflated — the gap between them is finding 2.

### Phase 1 findings at level-5 gen 256

`lock_predicate_audit_l5g256_20260901/` — 54 examples, 13,536 tokens,
thresholds fit on a disjoint 40% of examples from the same run (no cross-regime
transfer). The qualitative pattern replicates and one result sharpens.

| Operating point | savings | median lower | frac > 0.25 |
| --- | --- | --- | --- |
| premise A (lock at commit) | — | `0.514` | `96.8%` |
| earliest logged lock | `0.992` | `0.354` | `76.3%` |
| SureLock-style, `tau = 0` | `0.433` | `0.093` | `15.3%` |
| confidence-only, matched | `0.402` | `0.138` | `28.1%` |

**The headline result.** At `tau = 0` — the most conservative threshold a
KL gate can express, firing only where the adjacent-step posterior is
*numerically identical* — the rule still admits **71.0%** of committed tokens
(`7.5%` of all post-commit steps have exactly-zero KL). Those same tokens have
premise-A drift of median `0.514` with `96.8%` above `0.25`. So the posterior
can be bit-identical between consecutive steps while the hidden representation
keeps moving: **posterior stability and representational settlement are
separable, and separable in the regime that matters.** This is the strongest
available evidence for the A rung, and it is evidence *about the prior methods'
own admission signal*, not about naive confidence transfer.

Two differences from gen 64. The first-step drop is smaller (`96.8%` → `76.3%`,
versus `91.6%` → `55.0%` at gen 64), so drift is more sustained and less
front-loaded at long horizons. And the exchange rate is worse: reaching
`15.3%` provably-moving costs down to savings `0.433` here versus `0.548` at
gen 64.

**The upper bound is vacuous at gen 256** (`0.97`–`5.97`, versus `0.55`–`2.22`
at gen 64), because it sums ~125 step movements. Acquittal is therefore out of
reach in precisely the regime that matters, and a tighter certificate — direct
re-baselined drift logged at candidate lock steps — is needed before any rule
can be cleared rather than merely convicted.

## Phase 2 — Interventional (2026-09-02)

Does a stale reference actually change decoded output? Within-token pairs: the
same position frozen at the commit step (target) versus at a SureLock-style
gate's own lock step (control, `tau = 0`, i.e. the posterior is numerically
identical between steps). The gate is evaluated online during the baseline pass
so the cached row is snapshotted *at its own lock step*, not retroactively.
Candidates are sampled per example with a fixed seed, never drift-ranked.

| Directory | Contents | Role |
| --- | --- | --- |
| `phase2_rulelock_g64/` | group summary, pair summary, run config | gen 64, 269 pairs over 90 examples, 538 intervention decodes. |
| `phase2_pairs_g64_20260902/` | example-clustered bootstrap and sign tests | Paired deltas for the above. |
| `phase2_rulelock_l5_g256/` | group summary, pair summary, run config | level-5 gen 256, 108 pairs over 54 examples, 216 intervention decodes. |
| `phase2_pairs_l5g256_20260902/` | example-clustered bootstrap and sign tests | Paired deltas for the above. |

### Absolute harm, gen 64

| Arm | Non-target token changed | Answer text changed | Mean tokens changed |
| --- | --- | --- | --- |
| freeze at commit | `20.8%` | `6.7%` | `2.69` |
| freeze at gate lock (`tau = 0`) | `6.3%` | `0.7%` | `0.35` |

Median lock delay: `25` steps of 64.

### Paired deltas, gen 64 (269 pairs, 90 examples, 10k bootstrap)

| Endpoint | Delta | 95% CI | Sign test |
| --- | --- | --- | --- |
| non-target token changed | `+14.5` pp | `[+10.0, +19.3]` | `p = 2.8e-9` |
| non-target change count | `+2.34` tokens | `[+1.49, +3.26]` | `p = 1.3e-9` |
| normalized answer changed | `+5.9` pp | `[+3.0, +9.3]` | `p = 1.4e-4` |
| answer correctness changed | `+1.1` pp | `[0.0, +2.6]` | `p = 0.375` |

**Two findings.** The gate's delay buys real harm reduction — waiting for the
posterior to go numerically identical cuts output change from `20.8%` to
`6.3%`, and this is the strongest effect measured anywhere in this project
(`p ≈ 3e-9`). But **freezing at `tau = 0` is still not harmless**: about 1 in 16
interventions changes the decoded output even at the most conservative threshold
a KL gate can express. Movement does convert to harm.

**And the harm is token-level, not accuracy-level, at gen 64.** Answer
*correctness* moves `+1.1` pp with a CI touching zero (`p = 0.375`) — not
significant. Consistent with this repo's preserved strict-intervention null; do
not restate the token-level result as an accuracy result.

### Level-5 gen 256

`phase2_rulelock_l5_g256/` and `phase2_pairs_l5g256_20260902/` — 108 pairs over
54 examples, 216 intervention decodes, median lock delay `31` steps of 256.

| Arm | Non-target changed | Answer text changed | Correctness changed | Mean tokens changed |
| --- | --- | --- | --- | --- |
| freeze at commit | `37.0%` | `20.4%` | `1.9%` | `33.96` |
| freeze at gate lock (`tau = 0`) | `21.3%` | `8.3%` | `0.0%` | `12.42` |

| Endpoint | Delta | 95% CI | Sign test |
| --- | --- | --- | --- |
| non-target token changed | `+15.7` pp | `[+8.3, +24.1]` | `p = 9.1e-4` |
| non-target change count | `+21.5` tokens | `[+11.2, +33.0]` | `p = 9.0e-3` |
| normalized answer changed | `+12.0` pp | `[+4.6, +20.4]` | `p = 4.4e-3` |
| answer correctness changed | `+1.9` pp | `[0.0, +4.6]` | `p = 0.5` |

**The regime finding: the gate's protection does not scale.** The *paired
benefit* replicates almost exactly (`+15.7` pp here versus `+14.5` pp at gen 64),
but the *residual* harm at the gate's own operating point grows sharply —
`6.3%` → `21.3%` for non-target change, `0.7%` → `8.3%` for answer text, and
`0.35` → `12.42` tokens changed per intervention. At the generation lengths where
efficiency methods actually matter, freezing at the most conservative threshold a
KL gate can express still changes the output in **more than one intervention in
five**.

Correctness remains unresolved in both regimes (`p = 0.5` here, 2 discordant
pairs of 108) and is underpowered by design — do not read `0.0%` as safety.

## SureLock Clean-Premise Expansion (2026-09-06)

The clean `YF`-versus-`YC` analysis regrades saved generations with symbolic
MATH equivalence or isolated HumanEval execution and reports example-clustered
95% bootstrap intervals. Unlike the earlier summaries, it reports output
change, answer/program change, original accuracy, modified accuracy, net
accuracy delta, and both correctness-transition directions.

| Regime | Interventions | Downstream changed | Answer/program changed | Original acc. | Modified acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Levels 1-4, gen64 | 107 | `5.6%` | `2.8%` | `46.7%` | `46.7%` |
| Levels 1-4, gen256 | 108 | `14.8%` | `1.9%` | `46.3%` | `46.3%` |
| Level 5, gen64 | 108 | `7.4%` | `1.9%` | `9.3%` | `10.2%` |
| Level 5, gen256 | 108 | `21.3%` | `8.3%` | `7.4%` | `7.4%` |
| HumanEval, gen256 | 140 | `10.0%` | `10.0%` | `54.3%` | `55.0%` |
| AIME 2024, gen256 | 20 | `10.0%` | `5.0%` | `0.0%` | `0.0%` |

The 2x2 resolves a Level-5 horizon effect for downstream change (`+13.9` pp,
95% CI `[+2.8, +25.0]`) but not the difficulty-by-horizon interaction (`+4.6`
pp, `[-10.2, +19.4]`). HumanEval establishes non-math output sensitivity but
does not show correctness harm: one intervention changes failing code to
passing code and none change passing code to failing code.

Primary aggregate artifacts:

- `surelock_premise_suite_20260906/`
- `surelock_premise_2x2_analysis_20260906/`
- `../docs/surelock_premise_expanded_report_2026-09-06.md`

## LESS Admission-Reuse Audit (2026-09-06)

The released LLaDA LESS rule was integrated as a baseline sampler using
confidence `0.75`, top-8 coarsened JSD `0.04`, two stored prior winners, block
reset, all-accepted commitment, and immediate confidence fallback. Only genuine
LESS admissions, never fallback commitments, enter the freeze intervention.

| Dataset | Mean baseline NFEs | Interventions | Downstream changed | Answer/program changed | Original acc. | Modified acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Level-5 MATH, gen256 | `126.3 / 256` | 108 | `37.0%` | `20.4%` | `7.4%` | `7.4%` |
| HumanEval, gen256 | `49.7 / 256` | 140 | `16.4%` | `16.4%` | `58.6%` | `57.1%` |

All `108/108` Level-5 and `140/140` HumanEval interventions reproduce the
complete baseline prefix before the freeze: input tokens, raw top-1 predictions,
and LESS transfer masks. The target token never changes. Here "original"
means the normal LESS output checked against gold, while "modified" means the
frozen-row replay checked against gold; an answer can change while both versions
remain wrong.

LESS-accepted positions still move after commitment: median max drift is
`0.469` on 8,878 Level-5 body tokens and `0.448` on 10,035 HumanEval body
tokens; `91.9%` and `94.9%`, respectively, exceed `0.25`.

This is not a claim that published LESS is unsafe. LESS commits identity and
continues recomputation. The intervention asks whether its semantic admission
can also authorize reference freezing; the answer is no at the output level.
Accuracy harm remains unresolved.

The Level-5 result uses the complete held-out 54-example partition (records
80-133 of 134); HumanEval uses the tested first 70 of 164 tasks. Both sample two
eligible LESS admissions per example. See the report for metric definitions,
concrete wrong-to-wrong and passing-to-failing examples, and limitations.

Artifacts:

- `less_premise_l5g256_54x2_20260906/`
- `less_premise_reanalysis_l5g256_54x2_20260906/`
- `less_premise_humaneval_g256_70x2_20260906/`
- `less_premise_reanalysis_humaneval_g256_70x2_20260906/`
- `less_vs_surelock_premise_suite_20260906/`
- `../docs/less_premise_report_2026-09-06.md`

For definitions, split rules, and interpretation, read:

- `../docs/causal_robustness_protocol.md`
- `../docs/causal_robustness_results.md`

## Polestar-Cache Proxy Smoke (2026-09-07)

`polestar_cache_proxy_topkl_smoke_l5g256_4x1_20260907/` contains a
four-example Level-5/gen256 paired smoke of a token-level, per-head
attention-KL single-refresh proxy. It is explicitly not full Polestar-Cache.

All 5,824 selected refresh events were retained, and the strongest event per
example was tested. Full-prefix replay validity and target identity were 100%.
Neither the stale nor refreshed arm changed any downstream token (`0/4` each),
so the paired refresh benefit was zero. The current proxy therefore stops at
smoke rather than scaling to the full held-out cohort.

Detailed scope and limitations:
`../docs/polestar_cache_proxy_smoke_report_2026-09-07.md`.

## Official Elastic-Cache Evaluation (2026-09-07)

The official Elastic-Cache LLaDA generator at commit
`1960d8fc6231205a1ae4ebba3898d475e339f7e1` was compared with matched full
recomputation under the same model, prompts, confidence threshold, sliding
window, and EOS behavior.

| Dataset | Output changed | Answer/program changed | Baseline acc. | Elastic acc. | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Level-5 MATH, gen256 | 100.0% | 79.6% | 9.3% | 9.3% | 1.31x |
| HumanEval, gen256 | 65.7% | 64.3% | 58.6% | 57.1% | 1.49x |

Correctness transitions were balanced on Level-5 (2 correct-to-wrong and 2
wrong-to-correct) and nearly balanced on HumanEval (7 and 6). Aggregate accuracy
harm is unresolved, but exact-output and per-example correctness instability are
clear. The reported relative layer-recompute count is not a FLOP measurement.

Detailed report: `../docs/elastic_cache_official_report_2026-09-07.md`.
