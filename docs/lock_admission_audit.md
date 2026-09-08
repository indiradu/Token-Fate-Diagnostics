# Lock-Admission Audit Protocol

Established 2026-09-01. Asks whether published compute-lock methods for
diffusion LMs are entitled to the assumption that a locked position's reference
state has settled -- that is, whether the PhaseLock premises bind them or only
bind naive confidence-transfer commitment.

## Why premise A does not already answer this

Premise A measures post-commit drift baselined at **LLaDA's transfer step**
`s`: the step at which confidence decoding unmasks a position. Prior methods do
not lock there.

- **SureLock** locks an *already unmasked* position once its adjacent-step
  posterior KL indicates stability, then caches its K/V and skips its
  Q-projection and FFN rows. Its lock step `s' >= s`.
- **LESS** admits a commitment from confidence, top-1 persistence, and
  inter-step JSD — again evaluated over a window, so `s' >= s`.
- **TraceLock** learns from completed traces whether an intermediate token
  matches the final token. Semantic gate only; it does not by itself cache, so
  premises B and C do not bind it unless it is paired with reuse.
- **Polestar** already connects commitment and cache calibration through
  representation drift, so "drift exists" is not news against it. Only matched
  interventional harm is.

Evaluating any of these at `s` credits them with none of the settling that
happens over `[s, s']`. That would be a strawman, and a reviewer from the
SureLock camp would say so correctly. The audit therefore re-baselines drift at
each rule's own lock step.

## What each premise becomes under an admission rule `L`

| Premise | Question under `L` | Phase |
| --- | --- | --- |
| **A** | After `L` fires, is the reference state still moving beyond a meaningful tolerance? | 0, 1 |
| **B1** | Does `L` admit tokens where committing identity is harmless but freezing the reference is harmful? | 2 |
| **B2** | Does low observed past drift under `L` fail to certify future reference settlement? | 1, 2 |
| **C** | Within `L`'s admitted set, does reference-state drift predict freeze harm? | 2 |
| **U** | What does `L`'s admitted set save relative to a drift gate? | 1 |

If `L` already filters out high-drift tokens, C is moot for its admitted set.
That is a real possible outcome and vindicates the baseline; it is not a
failure of the audit.

The distinction between B1 and B2 matters. B1 is an intervention statement:
`YC` is the output when token identity is committed but representation is still
recomputed, and `YF` is the output when the same token is committed and its
reference is frozen. The clean failure is `YF != YC` while `YC = Y0`. B2 is an
online-certification statement: low drift over the previous few steps, or even
adjacent-step posterior identity, does not imply that all future K/V states will
remain inside a safe tolerance. If future reference states were known by oracle
to be identical to the frozen state, freezing would be safe in a conventional
transformer; the audit is about what online rules can infer without that oracle.

Raw drift is therefore not the full mechanism. A high-drift token can be
harmless if active tokens barely attend to it, while a smaller stale-reference
perturbation can matter if many active positions are sensitive to it. Treat
`K/V drift * attention exposure` as the simplest risk proxy and matched
interventional harm as the arbiter.

## Bounds, and which direction each one can argue

`drift_time_rows.csv` stores `d_t = ||h_t - b|| / ||b||` against the
post-commit base `b`. For a lock at observed index `i` with last observed index
`T`, in `||b||` units:

```
lower:  max_{j>i} | d_j - d_i |                                  (reverse triangle)
upper:  sum_{j>i} step_l2_drift_j * row_norm_{j-1} / base_norm   (triangle)
```

Both are renormalized by `row_norm_i / base_norm` to express movement against
the frozen reference's own scale, which is the quantity a staleness budget
would be written against.

**The asymmetry is the whole design.** The lower bound can only *convict*: a
large value proves movement continued past the lock. The upper bound can only
*acquit*: a small value proves movement was bounded. Reporting one alone would
let the conclusion be selected, so `analyze_lock_predicate_audit.py` always
emits both. `tests/test_lock_predicate_bounds.py` pins that the interval
brackets the movement actually realized on synthetic trajectories with known
ground truth — if that test fails, a conclusion can invert.

## Phases

**Phase 0 — `scripts/analyze_lock_admission_bound.py`.** Predicate-free, CPU,
no new decode. Computes the *best case for any conceivable rule*: minimum over
lock steps of the lower bound, reported against a savings budget
`f = (T - s') / (T - b)` so that locking at the last step (which saves nothing)
cannot win trivially. Result: at `f = 1.0` it reproduces premise A; by `f = 0.5`
the bound has decayed to `0.062` and is uninformative. **Phase 0 therefore
establishes that the free path cannot resolve delayed locks**, which is what
justifies the Phase 1 GPU spend.

**Phase 1 — `scripts/analyze_lock_predicate_audit.py`.** Observational.
Requires the post-commit logging added to `run_settlement_fate_audit.py` on
2026-09-01: `post_commit_posterior.csv` (`post_kl`, `post_jsd`,
`post_confidence`, `raw_top1`, `raw_top1_matches_commit`) plus
`step_l2_drift` / `row_norm` / `base_norm` in `drift_time_rows.csv`. These
signals were always computed at every position on every step and discarded for
unmasked ones.

**Phase 2 — `--intervention-mode rule_lock`.** Interventional: does a stale
reference actually change decoded output, or does the representation merely
move in directions attention is insensitive to?

The design is a **within-token** pair, which is tighter than the across-token
matching `choose_drift_candidates` needs. The same position is frozen twice:

| Arm | Freeze step | Reference row served |
| --- | --- | --- |
| target | commit step (`base_step`) | the commit-step row |
| control | the gate's own lock step | the row cached **at that step** |

Pairing the row to the step matters: freezing at the gate's step while serving
the commit-step row would misrepresent the method, which caches at its own lock
step rather than retroactively. The gate is therefore evaluated *online* during
the baseline pass (`--lock-rule-tau`, default `0.0` = posterior numerically
identical between steps) so the row can be snapshotted there.

Because both arms are the same token, `delta_non_target_changed` from the
existing `analyze_settlement_pairs.py` is exactly **the harm the gate's delay
avoided**, and the control arm's absolute rate is **the harm of freezing at the
gate's own operating point**. That second quantity is the one Phase 1 made
urgent: it is the saturation point where the posterior is identical yet 96.8% of
tokens still move.

Candidates are sampled per example with a fixed seed, never ranked by drift —
ranking by drift would preselect the tokens most likely to be harmed and inflate
both arms. `tests/test_rule_lock_selection.py` pins these invariants: pairs are
within-token, zero-delay tokens are dropped (two identical arms give no
contrast), semantically unsafe tokens are dropped, padding is excluded, and
selection is seeded rather than drift-ranked.

**Secondary arm (not run).** Rule-admitted versus rule-rejected, matched on
confidence and step, which is what tests premise B as an existence claim and
premise C within the admitted set. Deferred: each arm doubles GPU cost, and the
within-token design answers the sharper question first.

## Discipline specific to this audit

- **Compare at matched savings, never at matched threshold.** Thresholds are
  not commensurable across families. The common axis is the fraction of
  post-commit token-steps left frozen, the same row-update proxy
  `simulate_compute_utility.py` uses. Differences at matched savings are
  differences in *which* tokens a rule picks.
- **Freeze each threshold on a training run.** `--train-run` picks the
  threshold hitting a savings target on the train split; it is then applied
  unchanged to the reporting run. Sweeping on the reported run would tune a
  critique of someone else's method against its own outcome.
- **Never transfer thresholds across generation lengths.** The KL and JSD
  distributions shift with gen length, so a gen-64 threshold applied to a
  gen-256 run is not the same operating point and the rules stop being
  compared at matched savings. The level-5 gen-256 run has no same-regime
  training slice, so it uses `--self-split-train-frac` to fit thresholds on a
  disjoint subset of its own examples. `savings_drift_vs_train` reports how far
  realized savings moved from the target; a large value invalidates the matched
  comparison for that row.
- **Report the sweep, not a point.** A single operating point invites "you
  didn't tune their threshold as carefully as they did."
- **Name them "SureLock-style" / "LESS-style".** We reimplement the *admission
  rule* from the paper description, never the caching machinery, at our
  thresholds on our model. It is not a replication.
- **Phase 1 evaluates rules on the unfrozen trajectory.** That is the right
  counterfactual for "is the cached reference stale," but their caching would
  itself alter downstream posteriors. Phase 2's clamp is where intervention
  enters.
- **Exclude EOS/EOT.** Padding passes stability predicates trivially while
  drifting roughly twice as far as body tokens, so an admitted set contaminated
  with padding manufactures a conviction out of inert rows.
- **Run at gen 256 on level-5, not only gen 64.** At gen 64 the drift
  distribution is compressed and harm saturates above ~0.3; the same regime
  effect that hides premise C would hide this.
- **`raw_top1` must be read before the `torch.where` overwrite** in the decode
  loop, or it is just the committed token read back and persistence becomes
  vacuously true.

## Runs

Phase 1 baselines are `--skip-interventions` (observational) in `commit` mode,
which also emits `proposal_ledger.csv` for the pre-commit side at no extra
cost. Submission scripts: `cluster/phaselock_phase1_*.sbatch`.

| Slice | Data | Purpose |
| --- | --- | --- |
| `phase1_postcommit_g64_train` | MATH500 test, offset 0, limit 100, gen 64 | Freeze thresholds |
| `phase1_postcommit_g64_eval` | MATH500 test, offset 407, limit 90, gen 64 | Report |
| `phase1_postcommit_l5_g256` | `data/math500_level5.jsonl`, offset 80, limit 54, gen 256 | The regime where drift effects are visible |

Offsets are disjoint from each other and from the existing premise runs
(offsets 167 and 307 at gen 64; level-5 offsets 0 and 40).
