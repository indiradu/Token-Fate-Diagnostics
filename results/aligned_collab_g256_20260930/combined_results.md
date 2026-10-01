# All methods under the aligned protocol (LLaDA-8B-Instruct, 256 tokens, block 32, batch 1, 124 questions)

The three jobs each rerun the same B0. B0 output is token-identical across all three (124/124), and her dense
matches B0 on 124/124, so correct counts compare directly. Each speedup is against its own job's B0:
- aligned campaign: 16.028 s/example;
- Phase A: 15.535 s/example;
- learned: 15.930 s/example.

The Job column says which: `aligned` = `results/aligned_standard_g256_20260929`,
`phaseA` = `analysis_phaseA`, `learned` = `analysis_learned`. Intervals: 10,000-sample bootstrap over questions.

| Method | Whose | Job | Correct /124 | Δ correct vs B0 (pp) [95% CI] | Speedup vs B0 [95% CI] | Mean calls | Broken / fixed |
|---|---|---|---:|---:|---:|---:|---:|
| B0 (standard LLaDA) | — | all | 43 | — | — | 256 | — |
| Confidence-0.9 | ours | aligned | 44 | +0.81 [+0.00, +2.42] | +271.75% [+232.31, +324.21] | 67.9 | 0 / 1 |
| TraceLock-0.95 | ours | aligned | 43 | +0.00 [−7.26, +7.26] | +208.13% [+177.20, +248.14] | 79.6 | 10 / 10 |
| collab-TraceLock-LLaDA | hers (retrained) | learned | 41 | −1.61 [−8.87, +5.65] | +176.42% [+144.17, +219.40] | 89.6 | 12 / 10 |
| collab-open-confidence | hers (retrained) | learned | 42 | −0.81 [−5.65, +4.03] | +150.67% [+124.19, +185.10] (a) | 88.2 | 5 / 4 |
| collab-H2-guard90-delay1 | hers | phaseA | 47 | +3.23 [−4.03, +10.48] | +107.73% [+95.76, +121.36] | 143.3 | 8 / 12 |
| collab-safety-full | hers (retrained) | learned | 45 | +1.61 [−2.42, +6.45] | +100.57% [+94.21, +107.53] | 109.1 | 3 / 5 |
| SureLock-prompt-0.005 | ours | aligned | 45 | +1.61 [−4.84, +8.06] | +28.02% [+24.35, +31.87] | 256 | 8 / 10 |
| collab-SureLock-5e-3 | hers | phaseA | 34 | −7.26 [−14.52, +0.00] | +19.25% [+16.57, +22.11] | 256 | 15 / 6 |
| collab-SureLock-5e-4 | hers | phaseA | 34 | −7.26 [−14.52, −0.81] | +19.02% [+16.39, +21.82] | 256 | 14 / 5 |
| RepKL-freeze-0.0005 | ours | aligned | 43 | +0.00 [−5.65, +5.65] | +10.52% [+9.26, +11.75] | 256 | 6 / 6 |
| SureLock-gen-0.005 | ours | aligned | 44 | +0.81 [−4.84, +6.45] | +7.74% [+6.48, +8.96] | 256 | 6 / 7 |
| collab-capped-confidence | hers | phaseA | 43 | +0.00 [+0.00, +0.00] | −0.74% [−1.04, −0.43] | 244.9 | 0 / 0 |
| collab-dense | hers | phaseA | 43 | +0.00 [+0.00, +0.00] | −3.67% [−3.81, −3.54] | 256 | 0 / 0 |

(a) Open-confidence time is her additive estimate: A3 replay + controller time + extra Phase-2 planner time.
Her online Phase-1+2 planner produced identical outputs (124/124) in 5.958 s/example, which is +167.4%.

Declared adaptations for her methods:
- her code is from commit 24cd51b, with a transformers-5 loader shim, our prompts and our scorer;
- the learned models were retrained here with her recipe: 8,000 traces, seeded projection matching hers, controller
  max length 1024, safety threshold 0.808; they are not bit-identical to hers;
- open-confidence Phase 3 always abstains.
