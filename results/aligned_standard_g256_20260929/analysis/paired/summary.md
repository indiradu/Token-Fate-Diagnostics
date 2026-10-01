# Locking Policy Paired Analysis

- Source run: `/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/aligned_standard_g256_20260929/full`
- Examples: 124
- Policies: 6
- Repeats: 3
- Bootstrap reps: 10000
- Bootstrap unit: dataset/example_id paired clusters; repeats are not resampled independently

| dataset | comparison | quality n | accuracy delta | speedup % | sequence change | answer change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| humaneval | humaneval:policy_vs_B0:B0:vs:RepKL-freeze-0.0005 | 70 | 0.01429 | 11.13 | 0.6429 | 0.6429 |
| humaneval | humaneval:policy_vs_B0:B0:vs:SureLock-gen-0.005 | 70 | 0 | 8.489 | 0.6143 | 0.6143 |
| humaneval | humaneval:policy_vs_B0:B0:vs:SureLock-prompt-0.005 | 70 | 0.02857 | 28.42 | 0.7429 | 0.7286 |
| humaneval | humaneval:policy_vs_B0:B0:vs:TraceLock-0.95 | 70 | -0.01429 | 467.9 | 0.7857 | 0.7714 |
| humaneval | humaneval:policy_vs_B0:B0:vs:Confidence-0.9 | 70 | 0.01429 | 652.2 | 0.2429 | 0.2429 |
| math500 | math500:policy_vs_B0:B0:vs:RepKL-freeze-0.0005 | 54 | -0.01852 | 9.709 | 1 | 0.7963 |
| math500 | math500:policy_vs_B0:B0:vs:SureLock-gen-0.005 | 54 | 0.01852 | 6.769 | 1 | 0.7778 |
| math500 | math500:policy_vs_B0:B0:vs:SureLock-prompt-0.005 | 54 | 0 | 27.5 | 1 | 0.7037 |
| math500 | math500:policy_vs_B0:B0:vs:TraceLock-0.95 | 54 | 0.01852 | 91.99 | 1 | 0.7593 |
| math500 | math500:policy_vs_B0:B0:vs:Confidence-0.9 | 54 | 0 | 122.8 | 0.1667 | 0.07407 |
