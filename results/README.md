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

## Aligned Protocol Head-to-Head (2026-09-29 to 10-01)

All of our locking policies and all of the collaborator's methods (their commit `24cd51b`) were run under
one protocol: standard LLaDA decoding, 256 tokens, block 32, one token per step, batch 1, on 54 MATH
Level-5 and 70 HumanEval questions, scored with our scorer. Each GPU job reruns B0. B0 output matched
token for token across the three jobs (124/124 each), and the collaborator's dense decoder matched B0 on
124/124. Protocol: `../docs/aligned_protocol_2026-09-29.md`.

| Directory | Contents | Role in the study |
| --- | --- | --- |
| `aligned_standard_g256_20260929/` | `results.md`, policies, source gate, parity, pooled paired summary, FLOP summary, driver | Our six policies against B0. |
| `aligned_collab_g256_20260930/analysis_phaseA/` | `results.md`, parity, pooled statistics vs B0 and vs their dense | The collaborator's five non-learned methods. |
| `aligned_collab_g256_20260930/analysis_learned/` | `results.md`, parity, pooled statistics vs B0 | Their three learned methods, retrained here with their recipe. |
| `aligned_collab_g256_20260930/combined_results.md` | one table of all 13 methods | Cross-campaign comparison; each speedup uses its own job's B0. |
| `aligned_collab_g256_20260930/orchestration/` | compatibility shim, drivers, smoke gate, scorer bridge, sbatch files | How the collaborator's code was run and rescored. |

| Method | Whose | Correct /124 | Δ correct (pp) [95% CI] | Speed vs B0 |
| --- | --- | ---: | ---: | ---: |
| B0 | both | 43 | — | 1.00x |
| Confidence-0.9 (block) | ours | 44 | +0.81 [0.00, +2.42] | 3.72x |
| TraceLock-0.95 | ours | 43 | 0.00 [-7.26, +7.26] | 3.08x |
| TraceLock-LLaDA (retrained) | collaborator | 41 | -1.61 [-8.87, +5.65] | 2.76x |
| Open-confidence PhaseLock (retrained) | collaborator | 42 | -0.81 [-5.65, +4.03] | 2.51x (estimate) |
| H2 guard90_delay1 | collaborator | 47 | +3.23 [-4.03, +10.48] | 2.08x |
| Direct safety predictor (retrained) | collaborator | 45 | +1.61 [-2.42, +6.45] | 2.01x |
| SureLock-prompt 0.005 | ours | 45 | +1.61 [-4.84, +8.06] | 1.28x |
| SureLock 5e-3 | collaborator | 34 | -7.26 [-14.52, 0.00] | 1.19x |
| SureLock 5e-4 | collaborator | 34 | -7.26 [-14.52, -0.81] | 1.19x |
| RepKL-freeze 0.0005 | ours | 43 | 0.00 [-5.65, +5.65] | 1.11x |
| SureLock-gen 0.005 | ours | 44 | +0.81 [-4.84, +6.45] | 1.08x |
| Capped confidence | collaborator | 43 | 0.00 [0.00, 0.00] | 0.99x |
| Dense decoder | collaborator | 43 | 0.00 [0.00, 0.00] | 0.96x |

The open-confidence time is the collaborator's additive estimate; their online planner gave identical
outputs at 2.67x. Retrained models are not bit-identical to the collaborator's, and their Phase 3 always
abstains. Raw records and scored generations stay local.

## Selective cache and refresh campaigns (2026-11-04 to 2026-11-11)
Words: WORD LOCK = which blanks are written per call; REPRESENTATION CACHE = which rows' K/V are reused. LLaDA-8B-Instruct, bf16, temperature 0. Raw records, generations, frozen source copies, fitted-rule pickles and logs stay local; only summaries, pre-registrations and orchestration scripts are tracked. The highlighted cells are in `depth_refresh_20261109/README.md`.

| Directory | Contents | Role in the study |
| --- | --- | --- |
| `selective_cache_20261104/` | pre-registration, README, results tables (Part A staleness per layer and row class, Part C ablation) | The far-blank cache and what predicts stale rows; Elastic-Cache's trigger against a block-start schedule. |
| `farcache_stack_20261105/` | pre-registration, README, results, HumanEval diagnostics, Lctx attribution | The S32 and S16 caches stacked with C-0.9, C-0.8 and Lctx-0.8 on HumanEval, MATH and GSM8K. |
| `stale_cause_20261108/` | pre-registration with Outcome, noise-matched control, oracle repair | Same-size random noise does not reproduce the loss; no single row class restores the first divergence. |
| `learn2pd_control_20261108/` | official-code notes, results | Learn2PD word-lock baseline and the truncation confound. |
| `audit_crossline_20261108/` | blind audit of lost programs | Cross-line inconsistency is not a typical failure signature. |
| `depth_refresh_20261109/` | pre-registration with Outcomes, results of stages A to E (HumanEval half), extra metrics, orchestration | Blank-row partial refresh (P4) against dense refresh (K4) and the word-count trigger; MATH at 512 and 1024 tokens; word lock x cache grid; Elastic recipe in our engine. |
| `lctx_refit_20261110/` | pre-registration, fit reports, orchestration | Refit of Lctx for the 16-blank window and under the cache (decodes queued or running when written). |
| `lctx_features_20261110/` | pre-registration, orchestration | Gated feature ablation for Lctx (queued when written; no result). |
| `tracelock_stack_20261111/` | feasibility note, pre-registration, orchestration | TraceLock under the S32, K4 and P4 caches (queued when written; no result). |
| `metrics_survey_20261110/` | survey of the metrics 18 related papers report | Fetch-tool extractions, not verified against PDFs; basis for the comprehensive comparison. |

## Hand-over to the collaborator (2026-11-22)

| Directory | Contents | Role in the study |
| --- | --- | --- |
| `share_lctx_f3_20261122/` | `HANDOFF_LCTX_F3.md`, the Lctx-F3 rule pickle and token table (sha256 in the note), the collaborator's frozen engine `source_m1` with `engine_f3.patch` applied, loader, GPU check and its result | The new word lock for the word-lock sweep. The fitted pickle is tracked here, an exception to the rule above, because the collaborator runs it. Fitted on 60 MATH-500 L5 questions with λ set on 20 more, all disjoint from HumanEval-164, MATH-108, KodCode-held and GSM8K-held. |
| `lctx_features3_20261115/` | pre-registration with Outcome, offline feature code (`features3_lib.py`, `build_assets.py`, `fit_ablation3.py`), ablation table, token-table metadata | Where the F3 inputs (groups h, i, k, j) were defined and tested offline. Their gate failed (+1.23 share points over six, below the +5 asked for). `tests/test_lock_features3.py` checks the live decoder module against `features3_lib.py`. |

**Correction, 2026-11-23.** "MATH-108" in our comparison grid and word-lock sweeps is ours: `math_l5_eval54` + `data/math500_levels1to4_54_seed23.jsonl`. The collaborator's MATH-108 has a different level-5 half: 21 questions are shared, and 33 of the others are in Lctx-F3's fit (24) or threshold (9) split. So F3 is clean on our MATH-108 but not on theirs. `share_lctx_f3_20261122/HANDOFF_LCTX_F3.md` (Corrections) lists the IDs and scores the 75 questions clean on both sides. It also explains the `token_info` count mismatch (ids with a count above 0 vs summed counts; same file) and adds the KodCode-held / GSM8K-held results.
