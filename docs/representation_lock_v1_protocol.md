# PhaseLock representation lock v1

## Fixed semantic inputs

Representation experiments never regenerate a semantic decision inside A2.
For each example, A1 creates and saves one exact semantic commitment plan, and
every A2 representation arm replays that plan token-for-token and step-for-step.
The two retained semantic modes are defined in
`configs/semantic_lock_modes.json`:

- `safe_c095_s2`: the confirmed safety-first semantic baseline;
- `aggressive_c090_age4_s8`: the higher-speed Pareto mode, analyzed against
  its own A1 because it has known sequence-level semantic divergence.

All representation conclusions are based on paired `A1 -> A2` comparisons
within one semantic mode. An aggressive A1 difference from A0 is not counted as
representation-lock harm.

## Capped-drift algorithm

At denoising step `t`, let `C_t` be positions whose token identity has already
been semantically committed but whose reference is still live. For each
position, measure cosine distance between consecutive hidden states at the
configured observation layer:

`d_i(t) = 1 - cosine(h_i(t), h_i(t-1))`.

A position is eligible only when all three conditions hold:

1. `d_i(t) <= tau`;
2. drift has stayed below `tau` for `patience` consecutive observations;
3. at least `minimum_age` post-commit updates have been observed.

The per-step budget is

`B_t = ceil(representation_lock_fraction * |C_t|)`.

Among eligible candidates, v1 freezes at most `B_t`, ordered by smallest
current drift. With the initial fraction `0.06`, this is the representation
analogue of capped confidence: a gate decides admissibility, a score ranks safe
candidates, and a cap limits irreversible action. The signal itself remains
representation drift, not token confidence.

When selected, the position's current input and output hidden rows are frozen
at every transformer block. A2 remains dense, so active positions continue to
update while attending to those fixed references. The configured layer (8, 16,
or 24) is only where the online gate is measured; it does not mean that only one
layer is frozen.

## Sweep

LLaDA-8B has 32 transformer layers. The observation layers are 8, 16, and 24
to sample early, middle, and late representations without using only layer 24.

The full preregistered grid in
`configs/representation_lock_full_grid.json` is:

- drift threshold: `0.002, 0.005, 0.01, 0.02, 0.05`;
- patience: `1, 2, 3, 4`;
- minimum post-semantic age: `1, 2, 4`;
- observation layer: `8, 16, 24`;
- reference-lock fraction: `0.06`.

This is 180 representation arms per semantic mode. The first screen uses
`configs/representation_lock_coarse.json`: thresholds `0.002, 0.01, 0.05`,
patience `1, 2, 4`, age `1`, and all three layers (27 arms). The full grid is
run only after the smoke test validates exact plan replay and the coarse screen
locates the safety/coverage frontier.

The focused development set in
`configs/representation_lock_v1_shortlist.json` contains 12 capped-drift arms
covering every requested threshold, patience, age, and layer value. It also
includes one uncapped low-drift arm and one always-freeze-after-age-1 stress
control. These controls test whether the cap and representation gate are doing
real safety work rather than merely appearing conservative.

## Selection rule

A representation policy is eligible for confirmation only if:

- no A1-correct example becomes A2-incorrect in development;
- A1 and A2 have no normalized-answer change on math datasets and no execution
  correctness loss on HumanEval;
- any sequence divergence is reported, never hidden by unchanged accuracy;
- it has meaningful reference opportunity, measured as locked token-forwards
  divided by `NFE * generation_length`.

Among policies passing the safety gate, choose the Pareto frontier that
maximizes reference opportunity and earlier locking. A2 latency is diagnostic,
not the optimization target: compute savings are tested later by replaying the
exact A2 reference plan in A3.

## Logged evidence

Each run saves the resolved semantic and representation manifests, the exact
semantic and reference plans, threshold/patience/age/cap values, candidate and
eligible counts, per-lock drift rank, total reference locks, and time-weighted
reference opportunity. This is sufficient to distinguish a safe but too-late
gate from an unsafe aggressive gate and from a gate with no useful compute
headroom.
