# Related Papers for PhaseLock

Last checked: 2026-08-24.

This is a working literature map for PhaseLock. It should stay stricter than a
brainstorming note: claims below are tied to a primary paper page when available,
and weakly validated entries are marked as watchlist items.

## PhaseLock Axes

- Semantic lock: commit token identity while still allowing the representation
  to update.
- Reference lock: freeze the hidden/K/V reference exposed to other positions.
- Compute lock: remove a reference-frozen row from future row-wise compute only
  when the measured systems utility is positive.

The related-work distinction is:

- H-A: local or semantic stability is not the same as safe token commitment.
- H-B/H-C: safe token commitment is not the same as safe reference freezing.
- H-U: safe reference freezing is not the same as profitable compute removal.

## Citation Priority

### P0: Direct Competitors and Threats

1. **SureLock: Stopping Computation for Converged Tokens in Masked
   Diffusion-LM Decoding**  
   Source: arXiv:2602.06412, accepted ICLR 2026  
   Link: <https://arxiv.org/abs/2602.06412>  
   Pressure: the closest compute-lock baseline. It uses posterior stabilization
   to lock unmasked positions, cache K/V, and skip Q/FFN computation for locked
   rows.  
   PhaseLock stance: do not claim novelty for row-wise compute removal. The
   contribution is the separation of semantic-risk, reference-risk, and
   compute-utility gates. SureLock is a baseline and a reusable implementation
   substrate.

2. **TraceLock: The Path Matters: Learning a Token-Commitment Policy for
   Diffusion Language Models**  
   Source: arXiv:2605.24697  
   Link: <https://arxiv.org/abs/2605.24697>  
   Pressure: learns whether an intermediate token is likely to match the final
   token.  
   PhaseLock stance: learned future-token stability is a candidate semantic-risk
   estimator, not the full method. PhaseLock asks whether commitment is safe
   under intervention and then separately tests reference freezing.

3. **Ada-DLM: Towards Efficient and Effective Diffusion Language Model
   Inference via Semantic-Aware Adaptive Denoising**  
   Source: ACL 2026 long paper  
   Link: <https://aclanthology.org/2026.acl-long.819/>  
   Pressure: explicitly argues that scalar confidence criteria can miss semantic
   convergence.  
   PhaseLock stance: do not claim to be first on semantic convergence. The
   stronger claim is that semantic convergence must be tested as an
   interventional commitment-risk gate.

4. **LESS Is More: Mutual-Stability Sampling for Diffusion Language Models**  
   Source: arXiv:2606.16908  
   Link: <https://arxiv.org/abs/2606.16908>  
   Pressure: training-free sampler using confidence, top-1 persistence, and
   inter-step JSD to decide commitment.  
   PhaseLock stance: use LESS as a strong training-free stability baseline and
   as an input feature family for the semantic-risk gate.

5. **TACG: Trajectory-Aware Commit Gating for Diffusion Language Model
   Decoding**  
   Source: arXiv:2607.03236  
   Link: <https://arxiv.org/abs/2607.03236>  
   Pressure: trajectory-aware gate with history/persistence signals for
   commitment readiness.  
   PhaseLock stance: treat trajectory-aware signals as baselines or features;
   PhaseLock must show calibrated risk control and phase separation rather than
   another trajectory heuristic.

6. **Deferred Commitment Decoding for Diffusion Language Models**  
   Source: arXiv:2601.02076  
   Link: <https://arxiv.org/abs/2601.02076>  
   Pressure: shows that block boundaries and insufficient future context can
   make early commitment unsafe.  
   PhaseLock stance: important support for H-A. The official title does not
   include "with Confidence-Aware Sliding Windows"; that phrase should be used
   only as a method description.

7. **CoCommit: Don't Commit Alone: Joint Token Commitment in Diffusion Large
   Language Models**  
   Source: arXiv:2607.04469  
   Link: <https://arxiv.org/abs/2607.04469>  
   Pressure: argues that independently committing a bundle can miss conditional
   dependencies between selected positions.  
   PhaseLock stance: PhaseLock should not assume token-local risk is always
   sufficient. Keep a bundle-level extension or diagnostic in the plan.

8. **Polestar: Drift-Aware Cache Calibration and Token Commitment for Efficient
   Inference of Diffusion LLMs**  
   Source: arXiv:2607.14107  
   Link: <https://arxiv.org/abs/2607.14107>  
   Pressure: closest reference-risk threat because it connects token commitment
   and cache calibration through representation drift.  
   PhaseLock stance: the paper must empirically test whether drift predicts
   counterfactual reference-freeze harm under matched controls. Do not present
   "drift matters" as the novelty by itself.

9. **Window-Diffusion: Accelerating Diffusion Language Model Inference with
   Windowed Token Pruning and Caching**  
   Source: arXiv:2601.20332  
   Link: <https://arxiv.org/abs/2601.20332>  
   Pressure: already uses active/buffer/far-field computational token roles,
   pruning, and cached K/V.  
   PhaseLock stance: do not claim novelty from having multiple token states.
   Claim novelty from treating transitions as risk or utility gates.

### P1: Cache, Reference, and Row-Compute Baselines

10. **Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV
    Cache and Parallel Decoding**  
    Source: arXiv:2505.22618; ICLR 2026 status appears on project/repo pages  
    Link: <https://arxiv.org/abs/2505.22618>  
    Pressure: standard cache plus confidence-aware parallel decoding baseline.  
    PhaseLock stance: compare against it for practical speed-quality trade-offs.

11. **dKV-Cache: The Cache for Diffusion Language Models**  
    Source: arXiv:2505.15781; NeurIPS 2025  
    Link: <https://arxiv.org/abs/2505.15781>  
    Pressure: token-wise representation dynamics and delayed/conditioned KV
    caching.  
    PhaseLock stance: useful reference-freeze baseline; tests whether cache
    timing is enough without explicit intervention-risk estimates.

12. **dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive
    Caching**  
    Source: arXiv:2506.06295; accepted ICML 2026  
    Link: <https://arxiv.org/abs/2506.06295>  
    Pressure: long-interval prompt caching plus partial response updates guided
    by feature similarity.  
    PhaseLock stance: cache-refresh baseline for the reference-lock gate.

13. **d2Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching**  
    Source: arXiv:2509.23094; ICLR 2026  
    Link: <https://arxiv.org/abs/2509.23094>  
    Pressure: fine-grained selective K/V refresh using adaptive token selection.  
    PhaseLock stance: compare against as a strong reference-update baseline.

14. **FlashDLM: Accelerating Diffusion Language Model Inference via Efficient
    KV Caching and Guided Diffusion**  
    Source: arXiv:2505.21467; ICLR 2026  
    Link: <https://arxiv.org/abs/2505.21467>  
    Pressure: FreeCache reuses stable K/V projections; Guided Diffusion reduces
    denoising iterations with an auxiliary AR model.  
    PhaseLock stance: cite as a two-axis speedup baseline. "FreeCache" is a
    component name, not the formal paper title.

15. **DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token
    Selection and Partial Attention**  
    Source: official repo with ICML 2026 citation; refresh the OpenReview page
    before final BibTeX  
    Links: <https://github.com/scale-snu/DyLLM>,
    <https://openreview.net/forum?id=0azUrmsSyA>  
    Pressure: dynamic partial row computation rather than permanent locking.  
    PhaseLock stance: useful compute-utility comparator because it updates only
    salient rows instead of committing to monotonic removal.

16. **Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM
    Inference**  
    Source: arXiv:2510.09309  
    Link: <https://arxiv.org/abs/2510.09309>  
    Pressure: layer/head-aware cache eviction using mask-token attention.  
    PhaseLock stance: memory/reference-utility baseline, especially for long
    context experiments.

### P2: Adjacent Decoding and Scheduling Work

17. **Accelerating Diffusion LLMs via Adaptive Parallel Decoding**  
    Source: arXiv:2506.00413; NeurIPS 2025  
    Link: <https://arxiv.org/abs/2506.00413>  
    Pressure: combines dLLM marginals with an auxiliary AR joint model to choose
    how many tokens to sample in parallel.  
    PhaseLock stance: supports the point that local marginal confidence is not
    enough for safe joint commitment.

18. **Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models**  
    Source: arXiv:2506.19037; accepted ICML 2026  
    Link: <https://arxiv.org/abs/2506.19037>  
    Pressure: deterministic non-adjacent reveal groups reduce interactions among
    parallel unmasking decisions.  
    PhaseLock stance: scheduling baseline and possible post-filter for candidate
    bundles before semantic locking.

19. **Learning Unmasking Policies for Diffusion Language Models**  
    Source: arXiv:2512.09106; official Apple repo  
    Link: <https://arxiv.org/abs/2512.09106>  
    Pressure: frames unmasking as an MDP and learns a lightweight policy.  
    PhaseLock stance: relevant if PhaseLock learns a policy over semantic,
    reference, and compute states rather than independent thresholds.

20. **Diffusion Language Model Knows the Answer Before It Decodes**  
    Source: ICLR 2026 proceedings  
    Link: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/daadbff4d4ea884ca3d9d389a1dfc61c-Abstract-Conference.html>  
    Pressure: task-level answers can converge before full sequence refinement.  
    PhaseLock stance: distinguish token-level, sequence-level, and task-level
    safety metrics.

21. **Self Speculative Decoding for Diffusion Large Language Models**  
    Source: arXiv:2510.04147  
    Link: <https://arxiv.org/abs/2510.04147>  
    Pressure: exact-output acceleration control in evaluated settings.  
    PhaseLock stance: useful conceptual upper bound for zero-output-difference
    acceleration; PhaseLock may trade a controlled amount of output divergence
    for more compute reduction.

22. **Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large
    Language Models**  
    Source: arXiv:2508.00819; DAEDAL, accepted ICLR 2026  
    Link: <https://arxiv.org/abs/2508.00819>  
    Pressure: controls generated length rather than per-token compute state.  
    PhaseLock stance: orthogonal efficiency axis: DAEDAL changes the canvas
    length; PhaseLock changes the active compute set.

23. **DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding**  
    Source: ACL Findings 2026  
    Link: <https://aclanthology.org/2026.findings-acl.1048/>  
    Pressure: uses a diffusion LM as a parallel drafter for an AR verifier, with
    causal-consistency path search and adaptive draft length.  
    PhaseLock stance: adjacent speculative-decoding context, not a direct
    competitor unless PhaseLock adds an AR verifier.

### P3: Architecture, System, and Model Context

24. **Fast-dLLM v2: Efficient Block-Diffusion LLM**  
    Source: arXiv:2509.26328  
    Link: <https://arxiv.org/abs/2509.26328>  
    Relevance: block-diffusion training plus hierarchical block/sub-block caches.
    Useful context for deployment-oriented baselines.

25. **ReFusion: A Diffusion Large Language Model with Parallel Autoregressive
    Decoding**  
    Source: ICLR 2026 proceedings  
    Link: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/585979c057a1b30796cf317063559638-Abstract-Conference.html>  
    Relevance: moves parallelism from individual tokens to larger slots with
    AR infilling. Useful broader architecture context.

26. **FLARE: Diffusion for Hybrid Language Model**  
    Source: arXiv:2606.01774  
    Link: <https://arxiv.org/abs/2606.01774>  
    Relevance: converts hybrid-attention AR backbones into dLLMs with
    hardware-aware kernels and serving considerations.

27. **CDLM: Consistency Diffusion Language Models For Faster Sampling**  
    Source: arXiv:2511.19269  
    Link: <https://arxiv.org/abs/2511.19269>  
    Relevance: sampling/architecture-level acceleration; not a direct PhaseLock
    competitor unless the paper includes a broad efficiency related-work section.

28. **dInfer: An Efficient Inference Framework for Diffusion Language Models**  
    Source: arXiv:2510.08666  
    Link: <https://arxiv.org/abs/2510.08666>  
    Relevance: system framework separating model, diffusion iteration manager,
    decoding strategy, and cache manager. Important because FLOPs savings do not
    automatically imply latency savings.

29. **Sangam: Efficiently Serving Diffusion LLMs with the AR Stack**  
    Source: arXiv:2607.04206  
    Link: <https://arxiv.org/abs/2607.04206>  
    Relevance: serving system for cached dLLMs, with repeated prefill/decode
    scheduling issues. Important for PhaseLock's eventual wall-clock claims.

30. **Large Language Diffusion Models**  
    Source: arXiv:2502.09992; LLaDA  
    Link: <https://arxiv.org/abs/2502.09992>  
    Relevance: natural first backbone for PhaseLock experiments.

31. **Dream 7B: Diffusion Large Language Models**  
    Source: arXiv:2508.15487  
    Link: <https://arxiv.org/abs/2508.15487>  
    Relevance: second major open dLLM family; needed for cross-model
    generalization.

32. **Mercury: Ultra-Fast Language Models Based on Diffusion**  
    Source: arXiv:2506.17298  
    Link: <https://arxiv.org/abs/2506.17298>  
    Relevance: commercial/system speed-quality context, especially for code.
    Less useful as a mechanistic baseline.

## Watchlist and Citation Cautions

- **Accelerating Masked Diffusion Large Language Models: A Survey of Efficient
  Inference Techniques**: useful as a search aid, but during this pass I found
  only secondary/aggregator pages. Do not cite formally until a primary PDF,
  arXiv record, or proceedings page is found.
- **Venue labels for very recent papers** should be refreshed before submission.
  arXiv pages are stable enough for internal planning, but final BibTeX should
  use the latest venue/proceedings metadata.
- **Do not overclaim speedups** from related papers in the proposal. Use
  "reports" or "claims" unless reproducing the result locally.

## Clean PhaseLock Positioning

The safe novelty claim is not "we lock tokens," "we use semantic stability," or
"we cache stable states." Those are all crowded.

The sharper claim is:

> PhaseLock decomposes diffusion-LM acceleration into two causal risk decisions
> and one systems utility decision: token-identity commitment, reference-state
> freezing, and row-compute removal.

That framing gives the paper a falsifiable scientific question:

- Can semantic-commitment risk be controlled better than confidence or
  trajectory heuristics?
- Conditional on semantic commitment being safe, is reference freezing a
  separable source of harm?
- Conditional on reference freezing being safe, when does row removal actually
  improve wall-clock utility?
