# Causal Robustness Protocol

This protocol extends the initial token-fate causal pilot. Its purpose is to
test whether high-confidence predicted future regret identifies *downstream*
consequences, rather than merely the forced token's expected change.

## GSM8K High-Confidence Study

- Model: `GSAI-ML/LLaDA-8B-Instruct`.
- Training data for the selector: GSM8K training traces, examples 0-99.
- Held-out intervention data: GSM8K test traces, examples 100-599.
- Candidate slice: early (`block_t_frac <= 0.5`) positions whose confidence is
  at least the training 75th percentile (`0.7892530709505081`).
- Selector: a logistic token-fate probe trained only on the training slice,
  using entropy, margin, KL, top-1 flip, runlength, time, local mask ratio,
  and local context volatility. Confidence is excluded from the selector.
- Targets: one highest predicted-risk position per held-out example, for 500
  prospective targets. Future trace-regret labels are never used to select
  targets or controls.

### Same-Trajectory Controls

For each target, the control search is restricted to the same baseline decode
and the exact same global denoising step. A valid control must have lower
predicted risk, absolute confidence difference at most `0.05`, and relative
output-position distance at most `8`. The lowest-risk eligible position is
selected. Each trace row can appear once. The matching ledger is saved before
interventions; unmatched targets remain part of the target ledger but are not
included in paired estimates.

The selected calipers yielded 223 same-trajectory pairs in the pre-run ledger
audit. This count, and the achieved covariate gaps, must be reported alongside
all causal estimates.

### Intervention And Outcomes

At the selected denoising step, force the selected current top token to be
committed and complete the same 64-step decode. The primary endpoint excludes
the position that was directly forced:

\[
\mathrm{NonTargetChange}=\mathbf{1}[\exists j \ne i:\
x^{\mathrm{forced}}_j \ne x^{\mathrm{baseline}}_j].
\]

Secondary endpoints are the number and fraction of changed non-target tokens,
whether a later textual position changes, the target-token change, and exact
answer-correctness change. Baseline token sequences are reconstructed from the
trace's final-token labels and checked against the selected candidate label.

Paired effects are bootstrapped over examples/pairs, not dense token rows.

## Prediction Uncertainty

For high-confidence and early-high-confidence held-out slices, compare the
token-fate feature probe with a confidence-only isotonic baseline. Report
AUROC and average-precision differences with 95% bootstrap intervals obtained
by resampling held-out examples and retaining all of each sampled example's
token rows.

## Cross-Task Replication

The original MATH500 trace directories used offset 0 for both train and eval
and are therefore not used for this study. A replacement run uses MATH500 test
examples 0-99 for fitting and 100-199 for evaluation/intervention. The same
selector, matching, downstream endpoints, and example-level bootstrap are
applied after that disjoint trace job completes.
