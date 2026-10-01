# PhaseLock Locking Strategies

Prepared: 2026-09-09.

PhaseLock separates three actions: fixing a token's identity, freezing the
reference it exposes to other tokens, and skipping its future row computation.
The implemented controller currently uses a ranking policy for the first action,
a threshold policy for the second, and a configured execution backend for the
third. The research proposal extends this into two causal-risk gates followed by
a compute-utility gate.

This document explains the implemented rules, their timing, and the proposed
extension. It also records implementation limitations that affect how results
can be interpreted.

The implementation described here is pinned to local `main` commit
`c43be083840b1287ec35bb822e527c6aa4c5b461`. The working checkout when this document
was written was `pivot_efficient_compute`, at `4d16746`; the staged implementation
files are absent from that checkout. Source paths and line numbers below refer
to the pinned `main` commit unless explicitly described as proposal references.
The proposal is in [research_idea.md](research_idea.md), with a shorter explanation
in [the PhaseLock idea report](docs/phaselock_clear_report_2026-09-05.md).

**The policy at a glance.** Each action answers a different question.

| Action | Question | Implemented decision | Proposed decision |
| --- | --- | --- | --- |
| Token-identity lock | Which predictions should become fixed now? | Commit the top-ranked eligible positions under a per-step quota. | Admit positions whose estimated commitment risk is within a chosen budget. |
| Reference freeze | When should other positions see a fixed reference for this token? | By default, require small hidden-state drift for consecutive observations after commitment. | Admit positions whose estimated reference-freeze risk is within a separate budget. |
| Compute removal | When should the frozen row stop being recomputed? | In a sparse arm, skip the row after its scheduled reference freeze. | Skip only when the expected runtime benefit exceeds a specified minimum. |

There are three consequential limitations in the pinned implementation. The
persistence counter is not carried between steps; compute removal has no online
profitability estimator; and replay forces the first arm's token identities,
making final-output agreement insufficient evidence of freeze safety. These are
described in detail below. This document does not change the implementation.

**What is being locked.** Token identity is the discrete token stored in the
sequence, such as `7`. A hidden state is a vector computed for that position
inside a transformer block. Keys and values, or K/V, are the references that
attention queries use to obtain information from that position. Fixing the
discrete token does not stop its hidden state from changing as the surrounding
sequence develops.

In this repository, “semantic lock” means token-identity commitment. It is not a
claim that the meaning of the answer has converged. A different token sequence
may express the same answer, and an unchanged token can acquire a different
contextual contribution.

The intended progression is:

```text
masked, eligible position
    -> token identity committed; representation still recomputed
    -> reference frozen; dense reference arm still executes the row
    -> row computation skipped in the sparse arm; cached K/V remain available
```

The experiment runs the stages in separate matched arms rather than passing one
live execution through all three backends. In its sparse arm, reference freezing
and the decision to skip subsequent row computation share the same event.

| State | Can the stored token identity change through this runner? | Can its reference evolve? | Is its row computed inside the transformer blocks? |
| --- | --- | --- | --- |
| Masked and eligible | It can be replaced by a selected prediction. | Yes. | Yes. |
| Committed, not frozen | No; selection subsequently targets masked positions. | Yes. | Yes. |
| Frozen in the dense reference arm | No. | The saved block inputs/outputs are restored. | Yes, followed by restoration of the saved output. |
| Frozen in the sparse arm | No. | Saved K/V and block outputs are reused. | The frozen row is excluded from the wrapped blocks' active-row computations. |

Compute removal does not remove the token from the sequence or remove its
attention column. Active tokens can still attend to it.

**Signals available at each observation.** The runner performs a model forward
before making the next commitment and freeze decisions. The following quantities
are computed from that forward and previously observed forwards.

| Symbol | Meaning | Exact implementation detail |
| --- | --- | --- |
| `p_t(v)` | Current vocabulary distribution at a position | Softmax of the current logits, computed in float. |
| `c` | Confidence | Largest probability in `p_t`. |
| `m` | Margin | Largest minus second-largest probability. |
| `H` | Entropy | `-sum_v p_t(v) * log(p_t(v))`. |
| `D` | Adjacent-observation posterior drift | `KL(p_t || p_previous)`, clamped below at zero. |
| `f` | Top-1 flip indicator | One if the previous predicted token exists and differs from the current prediction; otherwise zero. |
| `r` | Intended prediction run length | Previous run length plus one when the prediction repeats; otherwise one. See the counter bug below. |
| `q` | Optional learned regret score | Positive-class probability from the supplied predictor bundle. Its interpretation depends on its training label. |
| `d` | Representation drift | `1 - cosine_similarity(h_current, h_previous)`, clamped below at zero. |
| `a` | Age since commitment | Number of subsequent executed model forwards observed after commitment. |
| `b` | Consecutive low-drift count | Number of successive eligible post-commit observations satisfying `d <= threshold`. |

On the first forward, the runner sets posterior KL to zero because no previous
distribution exists. The first representation drift is also zero for lack of a
previous hidden observation. These initial zeros do not constitute observed
stability. The default age and patience requirements prevent immediate freezing.

The learned scorer can additionally use entropy, normalized planned step index,
local mask ratio, and local average posterior KL. The local-window calculations
use the full sequence. Entropy is passed to the handwritten semantic priority
function but none of its handwritten formulas directly uses it.

Source: `scripts/run_phase_lock.py:53`, `:101`, `:312`;
`src/regret_remasking/phase_lock.py:45`.

**Token-identity commitment: eligibility and quantity.** The runner selects from
masked generated positions before the end of the current generation block. It
excludes prompt positions, already committed positions, and future-block
positions. Under normal progression, earlier blocks have already completed.
It does not first filter these candidates by an absolute confidence threshold.

At block entry, the ordinary decoding schedule distributes the initial masked
positions across the planned number of steps. Integer remainder transfers go
to the earlier steps. The resulting schedule supplies `base_k` at each step.
The PhaseLock runner then uses:

```text
N = number of currently eligible masked positions
alpha = semantic_lock_fraction
k = min(N, max(base_k, ceil(alpha * N)))
```

The default `alpha` is `0.06`. This is a fraction of the remaining eligible
positions, with a lower bound supplied by the ordinary schedule. It is not six
percent added on top of that schedule, and it is not a confidence cutoff.

For example, with `N = 32` and `base_k = 1`, the runner commits two positions:
`ceil(0.06 * 32) = 2`. With `N = 16` and the same quota, it commits one. If the
ordinary schedule requests four, it commits four, subject to availability.

The runner ranks candidates, chooses the top `k`, and writes their current
argmax token IDs into the sequence. Those positions are subsequently unmasked
and remain fixed. Their representations continue to be recomputed until a
reference freeze occurs. A block exits early when no masked positions remain,
which is how extra early commitments can reduce model-forward count.

Source: `src/regret_remasking/llada_trace.py:55`;
`scripts/run_phase_lock.py:213`, `:240`, `:266`, `:302`.

**Token-identity commitment: scoring policies.** Higher scores are selected
first. In the formulas below, `beta_KL = 1.0` and `beta_fate = 2.0` by default.

| Policy name | Priority score | Intended behavior |
| --- | --- | --- |
| `confidence` | `c` | Prefer the most confident current prediction. |
| `posterior`, alias `surelock` | `c * exp(-beta_KL * D)` | Discount confidence when the full distribution changed recently. |
| `margin` | `m` | Prefer a clear separation between the two leading alternatives. |
| `persistence` | `c * (1 + log(1 + r))` | Prefer confident predictions that have persisted. |
| `consensus` | `c * exp(-beta_KL * D) * (1 + 0.25*r) * (1 - 0.5*f)` | Combine confidence, posterior stability, persistence, and a penalty for a recent flip. |
| `token_fate`, aliases `trace`, `fate` | `c * exp(-beta_fate * q)` | Discount confidence when the supplied predictor estimates higher future disagreement. |

`consensus` is one combined score, not a vote across separately run models.
The `surelock` spelling is a local alias for the displayed expression; it does
not establish that this runner reproduces the complete published SureLock
method.

As a numerical illustration, suppose one commitment slot remains:

| Candidate | Confidence | Posterior KL | Confidence score | Posterior score, `beta_KL=1` |
| --- | ---: | ---: | ---: | ---: |
| A | 0.98 | 0.50 | 0.98 | approximately 0.5944 |
| B | 0.90 | 0.02 | 0.90 | approximately 0.8822 |

Confidence selects A; the posterior policy selects B. This illustrates a
ranking difference, not evidence that either commitment is harmless.

The default command-line experiment runs `confidence,posterior,consensus` as
separate semantic-policy arms. Although `PhaseLockPolicy()` itself defaults to
`confidence`, that object default is not the entire experiment matrix.

The learned policies require a predictor supplied through `--regret-model-path`.
Selecting a learned policy without a scorer raises an error. The scorer reads
the feature names stored in the bundle and uses `predict_proba(... )[:, 1]`.
The protocol expects training on disjoint traces; the runner does not prove
that the provided bundle satisfies that split requirement. A predictor trained
on future token disagreement is not automatically a predictor of intervention
harm or task correctness.

These priorities can still select low-confidence candidates because the quota
requires transfers. There is no implemented test of the form
`estimated_commitment_harm <= risk_budget` in this selection path.

Source: `src/regret_remasking/phase_lock.py:51`;
`src/regret_remasking/regret_model.py:94`;
`scripts/run_phase_lock.py:512`, `:561`.

**Persistence-counter limitation.** The formulas above describe the policy
functions, but the pinned runner initializes its `runlength` tensor to zero and
never replaces it with the updated value returned in the feature map. With the
runner's actual state updates, four identical consecutive predictions produce
feature run lengths `1, 1, 1, 1`, instead of `1, 2, 3, 4`.

Consequently, `persistence` applies the constant multiplier `1 + log(2)` to
confidence and has the same mathematical ranking as confidence, apart from
numerical/tie behavior. The persistence factor in `consensus` stays at `1.25`.
Its KL and flip factors still operate, so consensus does not generally collapse
to confidence. Results from this version must not be described as testing a
growing persistence bonus.

This was reproduced by executing the pinned `_feature_map` with repeated logits
and the runner's state updates, and confirmed by inspecting assignments in
`decode_one`. It remains unfixed in the implementation documented here.
Relevant locations: `scripts/run_phase_lock.py:76`, `:203`, `:418`.

**Reference freezing: the default decision.** The staged runner observes the
output of transformer block 24 by default, using one-based layer numbering.
For each committed, not-yet-frozen position, it compares that output with its
value on the previous executed forward:

```text
d_t = max(0, 1 - cosine_similarity(h_t, h_previous))

if d_t <= tau:
    b = b + 1
else:
    b = 0

freeze = committed AND not_already_frozen
         AND d_t <= tau
         AND b >= patience
         AND age >= minimum_age
```

The defaults are `tau = 0.02`, `patience = 2`, and `minimum_age = 1`. Both age and
the low-drift count reset to zero when the identity is committed. The new
commitment is not counted as a successful post-commit observation. Thus the
default gate first permits freezing after two subsequent qualifying forwards.

Age counts actual observed forwards. It is not necessarily the difference
between logged `global_step` values, because early completion can skip planned
steps before the next block starts. The counters persist across block boundaries.

The first post-commit drift comparison deserves care: its previous hidden state
came from the forward immediately before the token ID was written, when that
position was still masked. That comparison can reflect both the input-identity
change and changes in context. It is not a comparison of two exclusively
post-commit representations.

Small cosine drift at one observed layer is a proxy for stability. It does not
measure all layers' K/V changes, vector magnitude changes, or future changes
caused by newly unmasked context. The actual freeze snapshots block state across
all transformer blocks, although the gate measures one chosen layer.

Source: `scripts/run_phase_lock.py:128`, `:302`, `:312`, `:340`;
`scripts/run_four_arm_audit.py:307`.

**Reference freezing: presets and controls.** The named presets change the
drift tolerance and confirmation duration; they are not calibrated harm budgets.

| Preset | Maximum drift | Required consecutive observations | Minimum age |
| --- | ---: | ---: | ---: |
| `strict` | 0.005 | 4 | 1 |
| `default` | 0.020 | 2 | 1 |
| `loose` | 0.050 | 1 | 1 |
| `very_loose` | 0.100 | 1 | 1 |

The CLI defaults to `--representation-gate custom`, with explicit values equal
to the `default` preset. Choosing a named preset overrides the explicit
threshold, patience, and minimum-age arguments.

The following controls are implemented. All additionally require an eligible
committed, unfrozen position.

| Reference policy | Additional readiness conditions |
| --- | --- |
| `always` | `age >= minimum_age` |
| `confidence` | `c >= threshold` and sufficient age |
| `posterior`, alias `surelock` | `D <= threshold` and sufficient age |
| `drift`, alias `representation` | `d <= threshold`, low-drift count at least `patience`, and sufficient age |
| `drift_posterior` | The drift gate, plus `D <= threshold` |
| `drift_confidence` | The drift gate, plus `c >= 0.5` |

Confidence here is the current model's top-1 probability, not necessarily its
probability assigned to the token ID committed earlier. The stored token stays
fixed even if the model's current preferred token changes.

The confidence-only and posterior-only controls do not use drift patience.
The `runlength` argument is unused by the reference gate. The same numeric
`threshold` parameter changes meaning between policies: using `0.02` with the
confidence control means `confidence >= 0.02`, which is a very different rule
from `drift <= 0.02`. The combined posterior gate shares one threshold between
drift and KL; it does not expose separately tuned limits for the two quantities.

All listed presets prevent same-forward freezing. Custom settings such as
`minimum_age = 0` can allow immediate freezing for controls like `always`.
The delay is enforced through parameterized conditions, not an unconditional
exclusion of every newly committed position.

Source: `src/regret_remasking/phase_lock.py:18`, `:81`;
`scripts/run_phase_lock.py:517`, `:535`.

**A complete timing example.** This hypothetical example uses the default drift
gate. Here `F` denotes an executed forward, not the logged planned step index.

| Forward | Event or measured drift | Age | Low-drift count | Decision or execution behavior |
| --- | --- | ---: | ---: | --- |
| F | Token `7` is selected for commitment. | 0 | 0 | Write its identity; continue recomputing its state. |
| F+1 | Drift 0.015 | 1 | 1 | Wait for another qualifying observation. |
| F+2 | Drift 0.025 | 2 | 0 | Reset the count because 0.025 exceeds 0.02. |
| F+3 | Drift 0.018 | 3 | 1 | Continue waiting. |
| F+4 | Drift 0.010 | 4 | 2 | Snapshot and freeze its reference. |
| F+5 | The reference is already frozen. | — | — | During F+5, dense P2 restores it and sparse P3 skips its row computation. |

The freeze cannot save work in the forward that just produced its snapshot.
Skipping starts on the next executed forward. If decoding has already finished,
there is no subsequent work for that late freeze to save. The runtime provides
no unfreeze transition within a decode.

**Compute removal: what the backends actually do.** Backend selection is an
experiment configuration. It is not currently a learned decision made separately
for every token.

| Backend or arm | Reference handling | Computation performed |
| --- | --- | --- |
| Dense P1 | References remain live. | Ordinary dense block computation. |
| P2, named `kv_cache` | Save and restore each frozen row's block inputs and outputs. | Ordinary dense blocks still process all rows. |
| P3 `row_sparse` | Store frozen K/V and outputs; assemble full-length K/V workspaces. | Run the wrapped blocks' projections, active-query attention, attention-output work, and FFNs on active rows. |
| P3 `row_sparse_packed` | Reuse persistent full-length K/V workspaces and update active positions. | Use the same lock schedule and active-row computation with different workspace handling. |

P2's `kv_cache` name is shorthand for a dense frozen-reference control. This
backend does not store actual K/V tensors: it clamps block inputs and outputs,
so K/V derived from frozen inputs are held fixed under deterministic execution.
P3 explicitly stores and reuses K/V tensors.

P3 computes active queries against K/V for the full sequence, including frozen
positions. It preserves the original position indices for rotary embeddings.
Prompt rows and future masked rows remain active. Full-length state storage and
the dense attention primitive remain; operations outside the wrapped blocks are
not automatically made sparse. The implementation supports one example per
forward and rejects a conventional past-KV-cache input in the sparse backend.

There is no call to a runtime-gain predictor, no break-even threshold, and no
choice to retain a frozen row's computation because skipping is predicted to be
slower. P3 removes its row from subsequent active sets once the reference-lock
event occurs. A sparse implementation can therefore reduce row work and still
increase latency through packing, allocation, copying, or Python overhead.

The legacy `scripts/run_four_arm_audit.py` driver freezes immediately on its
semantic-commit events. The staged `scripts/run_phase_lock.py` driver reuses its
`ReferenceRuntime` backend but adds the delayed gate described above. The two
drivers do not have identical admission policies.

Source: `scripts/run_four_arm_audit.py:106`, `:140`, `:182`, `:206`, `:252`,
`:307`, `:470`; `scripts/run_phase_lock.py:175`, `:371`, `:626`.

**How the experiment combines policies.** The runner first generates A0, then
loops over semantic policies, reference policies, and requested sparse backends.
The default settings produce A0 plus three P1, three P2, and three P3 arms:
ten arms per example.

| Arm | Semantic decisions | Reference decisions | Intended comparison |
| --- | --- | --- | --- |
| A0 | Ordinary confidence schedule; extra lock fraction zero. | None. | Dense baseline. |
| P1 | Selected semantic priority with the configured commitment quota. | None. | A0 versus P1 examines the changed commitment schedule and policy together. |
| P2 | Replay P1 positions, steps, and token IDs. | Apply the selected reference gate. | Examine reference freezing under a fixed semantic plan. |
| P3 | Replay the same P1 semantic plan. | Replay P2 reference-lock positions and steps. | Examine sparse execution under matched decision schedules. |

Requesting `row_sparse` or `row_sparse_packed` automatically causes P2 to run,
because P3 needs its reference plan. Requesting only `kv_cache` runs P2 but no
P3. Although `dense` is an accepted compute-policy spelling, requesting it alone
produces the always-run A0/P1 arms; it does not create a separate P2 freeze arm.

P3 replays P2's reference positions and times, not P2's tensor snapshots. Each
arm snapshots its own states when `runtime.commit()` is called. Matching event
schedules therefore does not guarantee that the saved tensors are identical.

Source: `scripts/run_phase_lock.py:258`, `:357`, `:575`, `:595`, `:599`, `:639`.

**Why matching replay outputs cannot establish freeze safety.** During semantic
replay, the runner explicitly overwrites a selected position's current predicted
token with the `token_id` stored in P1's plan, then writes it into the sequence.
If the plan covers every generated position and the replay completes, P1, P2,
and P3 end with the same prescribed token identities even if their live model
predictions differ substantially.

Thus final sequence or answer agreement in these matched replay arms is largely
fixed by construction. It cannot demonstrate that reference freezing would
preserve a freely continuing decoder's answer. It also cannot independently
validate numerical equivalence between dense and sparse backends. A0 versus P1
does not have this same forced-replay relationship, although it jointly changes
selection policy and commitment schedule.

For a backend-equivalence claim, the relevant additional evidence would compare
live logits, active-row hidden outputs, and cached references under the same
input and event history, with an explicit numerical tolerance. For a causal
quality claim, an evaluation must define which interventions are held fixed and
which downstream predictions remain free to respond. The pinned staged runner
does not supply that free-continuation test merely by reporting final answers.

This qualifies the output-preservation interpretation in the pinned
`docs/phase_lock_screening_results.md`. That file contains historical screening
observations; those observations do not override the replay construction above.

Source: `scripts/run_phase_lock.py:258`, especially `:264`, and `:304`.

**The proposed policy beyond these heuristics.** The research design has a cheap
candidate-routing step followed by three action-specific gates. Candidate routing
is preparation for the first lock, not a fourth lock. It concentrates additional
risk estimation on apparently stable, high-confidence positions. The current
quota-based runner does not implement that preliminary hard filter.

Use these outcome definitions from the proposal:

```text
Y0 = continuation under the specified baseline
YC = continuation after committing a token identity, with live representations
YF = continuation with that identity fixed and its reference frozen
YR = continuation with the reference frozen and its row computation removed
```

For the first lock, the proposed estimator is:

```text
r_commit(i,t) = P(YC != Y0 | observed history through t)
commit only if r_commit(i,t) <= epsilon_commit
```

For the second lock, it is:

```text
r_reference(i,t) = P(YF != YC | observed history, commitment admitted)
freeze only if r_reference(i,t) <= epsilon_reference
               and age >= minimum_reference_age
```

The event behind each probability must be defined before fitting or evaluating
the estimator. Any downstream token change, extracted-answer change, and a
correct-to-incorrect transition are different targets. A change-probability
estimate cannot be relabeled as accuracy-harm probability without changing and
validating its label.

The proposal discusses reference drift combined with exposure to active-token
attention as a possible signal. That is a proposed estimator input, not an
implemented formula in `representation_ready`. A small drift threshold alone
does not produce a calibrated value of `r_reference`.

The third gate asks whether compute removal is useful on the actual runtime:

```text
g_compute = latency_with_reference_frozen - latency_with_row_removed

remove row only if its reference has been admitted for freezing
                   and E[g_compute | runtime state] >= g_min
```

Runtime state can include active-row count, sequence length, batch size, device,
cache layout, backend, and remaining forward opportunities. The current runner
does not estimate this conditional gain or apply `g_min`. Output equivalence
under a valid frozen reference is a condition to verify separately from speed.

The proposal does not provide fitted risk estimators, validated risk budgets,
or a hardware-specific break-even rule through the existing default flags.
In particular, `semantic_lock_fraction` is not `epsilon_commit`, and the drift
threshold `0.02` is not a two-percent causal-risk budget.

Proposal references: `research_idea.md:388` for candidate routing,
`:414` for commitment risk, `:433` for reference risk, `:454` for compute utility,
and `:476` for the compact policy sketch, as inspected on `4d16746`.

**Configuration and recorded evidence.** These are the default settings of the
pinned staged CLI, not empirically established optimal settings.

| Setting | Default |
| --- | --- |
| Semantic policies | `confidence,posterior,consensus` |
| Semantic lock fraction | `0.06` |
| KL penalty weight | `1.0` |
| Learned-regret penalty weight | `2.0` |
| Reference policy | `drift` |
| Measurement layer | `24` |
| Reference gate selector | `custom` |
| Drift threshold / patience / minimum age | `0.02 / 2 / 1` |
| Compute policy | `row_sparse` |
| Planned denoising steps / generated length / block length | `64 / 64 / 32` |
| Model / dtype / seed | `GSAI-ML/LLaDA-8B-Instruct / bf16 / 17` |

The runner records `run_config.json`, per-arm JSONL outputs,
`phase_ledger.jsonl`, and `phase_plans.jsonl`. Semantic events include selected
position, token ID, score, confidence, and step. Reference events record the
measurement layer, drift, threshold, count, age, and position. The plan file
records the event schedules used for replay.

There is no separate `compute_lock` event. Compute exclusion is inferred from
the chosen sparse backend and the replayed reference-lock events.

The metrics need careful interpretation:

| Recorded quantity | What it measures | What it does not establish |
| --- | --- | --- |
| `nfe` | Number of executed model forwards. | FLOPs or runtime saved within each forward. |
| `masked_token_forwards` | Sum of masked generated positions before forwards. | Active-row count; it omits prompt rows and committed but still-active rows. |
| `reference_lock_count` | Number of positions admitted to reference freezing. | Freeze safety or positive runtime benefit. |
| `latency_s` | Instrumented decoding duration, including controller work and logging. | Isolated kernel speed or a calibrated online utility estimate. |
| Final sequence / answer comparisons | Differences between saved arm outputs. | Independent freeze-safety evidence when token IDs are forcibly replayed. |

The runner synchronizes CUDA before completing its recorded decode duration.
Its timing region also includes feature computation, Python work, ledger writes,
and final output preparation. There is a dense model warmup; this should not be
described as a separately warmed benchmark of each sparse backend.

Source: `scripts/run_phase_lock.py:206`, `:223`, `:288`, `:374`, `:427`, `:476`;
`scripts/analyze_phase_lock.py:166`.

**Source map and verification scope.** The following files define the behavior
documented here. Read the pinned revision when checking a claim, since the
current checkout may contain a different implementation or lack these files.

| Source | Responsibility |
| --- | --- |
| `src/regret_remasking/phase_lock.py` | Policy defaults, semantic scores, reference gates, and gate presets. |
| `scripts/run_phase_lock.py` | Feature collection, eligibility, quotas, state updates, staged replay, CLI, and logging. |
| `scripts/run_four_arm_audit.py` | Dense reference restoration and sparse execution; also an earlier standalone audit driver. |
| `src/regret_remasking/llada_trace.py` | Ordinary token-transfer schedule and shared model helpers. |
| `src/regret_remasking/regret_model.py` | Loading and evaluating the optional learned scorer. |
| `scripts/analyze_phase_lock.py` | Paired output and runtime summaries. |
| `docs/phase_lock_algorithm.md` | Earlier staged design and preset descriptions. |
| `docs/phase_lock_screening_results.md` | Historical screens, subject to the replay interpretation limitation above. |

For example, this read-only command displays the policy definitions without
switching branches:

```bash
git show c43be083840b1287ec35bb822e527c6aa4c5b461:src/regret_remasking/phase_lock.py
```

The document was checked against the pinned source, including policy formulas,
defaults, event timing, and replay assignments. Small local checks verify the
worked scoring/quota examples, drift-counter example, and the reported
persistence behavior. No LLaDA decoding experiment, GPU performance measurement,
source-code fix, or new causal validation is claimed by this documentation work.
