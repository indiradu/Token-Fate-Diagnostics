As an ICML idea in August 2026, I would currently rate this 6/10 as written: borderline, with a plausible path to 8/10+ if you sharpen the contribution.

The underlying observation is good. Your own evidence says the learned trace predictor has its residual advantage specifically among high-confidence tokens, where it improves over confidence by +0.0174 AUROC and +0.0124 AP on held-out GSM8K. That directly targets the dangerous case for irreversible locking: “looks certain now, changes later.” And the proposed algorithm is coherent: cheap confidence screening first, learned fate gating only for plausible lock candidates, followed by separate commit and compute locks.

My ICML-style scorecard
Dimension	Score	Assessment
Importance	8/10	dLLM inference efficiency is important and timely
Core intuition	8/10	“confidence ≠ future stability” is compelling
Novelty as written	5/10	serious overlap with several 2026 papers
Technical depth	6/10	currently mostly gating + engineering
Experimental falsifiability	9/10	very clear hypotheses and Pareto comparisons
Potential impact	8/10	high if it actually moves quality–latency frontier
Current ICML probability	~35–45%	borderline
After strong reframing/results	~65–75%	credible ICML paper

The biggest problem is not SureLock anymore.

The dangerous related work is TraceLock

TraceLock is extremely close to the conceptual center of your proposal. It explicitly trains a lightweight controller using future-stability labels obtained from completed diffusion traces and asks whether an intermediate proposed token agrees with the final token. It then uses this prediction to decide whether the token should become irreversible.

Its formulation is almost:

trace state→P(future stable)→commit / revise.

Your current formulation is:

high confidence→P(future regret)→commit / keep active.

That distinction is real, but at present it is not large enough to carry an ICML paper by itself.

Worse, TraceLock is already explicitly arguing that its learned decisions capture information beyond scalar confidence. So I would not make the primary claim:

“A learned token-fate predictor can predict whether confident tokens remain stable.”

A reviewer can immediately respond:

“TraceLock already learns future stability from traces and uses it for commitment.”

There is actually an earlier learned baseline in that lineage too: TraceLock describes Learn2PD as learning a token filter from agreement with the final decoded sequence.

There is also a rapidly filling “trajectory-aware commitment” literature

TACG is even closer to your motivating failure case. Its central argument is that a transient high-confidence top-1 prediction can be unsafe to commit, and it uses trajectory history to decide commitment readiness.

SWD likewise attacks premature commitment using temporal stability/KL rather than instantaneous confidence.

LESS combines confidence, top-1 persistence, and inter-step JSD as a joint criterion for safe commitment.

And Polestar jointly addresses token commitment and compute/cache reuse using representation drift, reporting an accuracy-throughput Pareto improvement.

Meanwhile SureLock has already established the systems-level mechanism you want to exploit: locking converged positions, caching K/V, and skipping subsequent Q/FFN computation, with reported 30–50% algorithmic FLOP savings on LLaDA-8B.

So the reviewer landscape now looks something like:

SureLock
→ stable token → stop compute

TraceLock / Learn2PD
→ learned future agreement → commit

TACG / SWD / LESS
→ temporal trajectory → safer commitment

Polestar
→ drift → commitment + cache efficiency

Your current proposal
→ learned future-regret predictor, selectively applied to high-confidence candidates → commit → compute lock

That final row is distinguishable, but it needs a stronger scientific principle.

The part I think is genuinely promising

I would center the paper on selective risk control, not token-fate prediction.

Your best observation is essentially:

Value of learned prediction

≈uniform across tokens.

Instead,

Value of learned prediction≫among apparently safe lock candidates.

That is more interesting than “learn a better commitment score.”

You already articulate this correctly in the draft: the learned model should operate only where confidence is misleading, while low-confidence positions need no expensive classifier.

I would turn that into a decision-theoretic problem.

Instead of:

predict token fate and threshold it

formulate:

π
max
	​

E[FLOPs saved by locking]

subject to

Pr(premature lock∣π)≤ϵ.

Or even:

π
max
	​

E[compute saved]−λE[future lock regret].

Then the learned fate model is not the contribution. It is an estimator of conditional lock risk.

That immediately gives the paper more ICML flavor.

The stronger paper I would pursue
Risk-Controlled Selective Locking for Diffusion Language Models

The central claim becomes:

Existing methods detect apparent convergence. We instead treat irreversible token locking as a selective prediction problem: lock only when the estimated future-regret risk is below a calibrated risk budget, and spend prediction overhead only on candidates for which confidence is insufficient.

Then introduce three levels:

ACTIVE→COMMITTED→COMPUTE-FROZEN.

And crucially, optimize each transition under a different risk criterion.

For example:

r
i
commit
	​

=P(x
i,t
	​


=x
i,T
	​

∣z
i,≤t
	​

),

while compute-lock risk could be

r
i
compute
	​

=P(stale representation materially alters other tokens∣z
i,≤t
	​

).

That second quantity is much more interesting.

Right now your proposal effectively assumes:

if token identity is stable, its representation is safe to cache.

But those are not the same claim.

A token can satisfy

x
i,t
	​

=x
i,T
	​


while its hidden/K/V representation continues changing enough to affect neighboring positions.

Your draft already implicitly recognizes this by separating commit lock from compute lock and warning that stale representations can hurt neighbors.

I would promote that observation from an implementation detail to a central research question.

That could give you a substantially stronger novelty triangle
1. Semantic fate

Will this token identity eventually change?

2. Representational fate

Even if its identity does not change, will its representation continue to change materially?

3. Computational fate

Can we remove this token from future computation without affecting the final sequence/task result?

Then show empirically that

semantic convergence

=representational convergence

=computational dispensability.

That is potentially an 8.5/10 ICML idea.

It goes beyond TraceLock, because TraceLock principally asks when a candidate can be committed.

It goes beyond SureLock, because SureLock uses local posterior stability as its locking criterion.

And you get a direct scientific question rather than just another decoder heuristic.

What would cause me to reject the current version?

If I were Reviewer 2, my review would probably say:

The paper identifies high-confidence false commitments and proposes adding a learned future-stability gate before locking. However, recent approaches including TraceLock already learn future stability from decoding traces, while TACG/SWD/LESS explicitly leverage temporal signals to prevent premature commitment. SureLock and Polestar additionally address computation/cache reuse. The principal novelty therefore appears to be restricting the learned controller to high-confidence candidates and composing it with compute locking, which may be incremental.

That is the central danger.

And your current +0.017 AUROC / +0.012 AP result is interesting diagnostic evidence, but nowhere near enough by itself to answer that criticism.

AUROC is not what ultimately matters.

The quantity reviewers will care about is something like:

Δaccuracyat fixed FLOPs/latency

or

ΔFLOPsat fixed accuracy.
The experiment that would make me excited

Forget broad predictor AUROC as the headline.

Construct the high-confidence disagreement set:

H={(i,t):c
i,t
	​

>τ
c
	​

,but token eventually changes}.

Then ask:

Can your method find the rare dangerous tokens?

For every method:

Confidence
KL
confidence + KL
SWD
TACG
LESS
TraceLock
SureLock criterion
your selective fate model
oracle

measure:

P(future change∣locked)

against

fraction of token-step FLOPs eliminated.

That single Pareto plot could become the paper.

Then do the actual end-to-end plot:

x-axis: wall-clock speedup
y-axis: task quality

with LLaDA + Dream + ideally another backbone.

Your draft is already headed toward the appropriate matched-budget evaluation.

One change I would make immediately

Do not phrase the novelty as:

“Token fate tells us whether a high-confidence token will stay the same.”

TraceLock makes that framing very difficult to defend.

Instead:

Locking is a risk-sensitive decision rather than a convergence-detection problem. Existing signals tell us how settled a token appears; we estimate the conditional cost of making that state irreversible, and invoke that estimator only where cheap convergence signals become unreliable.

Even better, if the experiments support it:

Semantic convergence is insufficient for compute convergence.

That is a much more defensible research thesis.

Bottom line

Current document: 6/10 ICML idea, borderline.

Before the 2026 TraceLock/TACG/Polestar wave: I probably would have called it 8/10.

With only “confidence + learned fate gate + SureLock”: likely too incremental for a comfortable ICML accept.

With risk-controlled selective locking + explicit semantic/representational/compute fate distinction + real end-to-end Pareto gains: 8–8.5/10, and I would consider that a serious ICML project.

The best part of your current draft is therefore not the classifier. It is the emerging idea that different tokens deserve different amounts of evidence before an irreversible compute decision, and that commitment safety and compute safety may be two different prediction problems. That is where I would push the work.

I can also monitor new dLLM-decoding papers, because this niche is moving unusually fast and another overlapping preprint could materially change the novelty position.