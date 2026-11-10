# Part C results: which ingredient buys the speed (gen 512, MATH, 54 questions: 27 L5 + 27 L1-4; one process, one timed repeat)

Speeds are ratios of summed decode seconds inside this one process (paired bootstrap over questions, 10,000 resamples, 95% interval). Answers: lost = correct under the reference and wrong under the arm; gained the reverse; net = gained - lost. Resolution about +-4 answers at 54 questions.

| cell | word lock | speed vs Conf-0.9 [CI] | speed vs SPARSE-none [CI] | speed vs W16-dense [CI] | calls | ms/call | correct | net vs Conf-0.9 [CI] (lost/gained) | net vs W16-dense [CI] (lost/gained) | changed text vs Conf-0.9 | fills budget | rows computed vs Conf-0.9 | refreshes / dense calls per question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Conf-0.9 | block-32 Conf-0.9 | - | 1.094 [1.091, 1.096] | 0.981 [0.912, 1.038] | 159.1 | 85.2 | 15/54 | - | -1 [-6, +3] (3/2) | 0.00 | 0.61 | 1.000 | 0.0 / 0.0 |
| SPARSE-none | block-32 Conf-0.9 | 0.914 [0.912, 0.917] | - | 0.897 [0.835, 0.948] | 159.1 | 93.1 | 15/54 | +0 [+0, +0] (0/0) | -1 [-6, +3] (3/2) | 0.00 | 0.61 | 1.000 | 0.0 / 0.0 |
| W16-dense | window-16 Conf-0.9 + EOS truncation | 1.020 [0.964, 1.096] | 1.115 [1.055, 1.198] | - | 156.8 | 84.7 | 16/54 | +1 [-3, +6] (2/3) | - | 0.93 | 0.67 | 0.977 | 0.0 / 156.8 |
| W16-cache-block | window-16 Conf-0.9 | 1.878 [1.734, 2.024] | 2.054 [1.898, 2.212] | 1.842 [1.693, 1.991] | 186.4 | 38.7 | 14/54 | -1 [-7, +5] (5/4) | -2 [-8, +3] (5/3) | 1.00 | 0.52 | 0.105 | 0.0 / 11.7 |
| W16-cache-block+EOS | window-16 Conf-0.9 + EOS truncation | 2.012 [1.891, 2.142] | 2.200 [2.068, 2.342] | 1.973 [1.844, 2.107] | 173.9 | 38.7 | 14/54 | -1 [-7, +5] (5/4) | -2 [-8, +3] (5/3) | 1.00 | 0.52 | 0.098 | 0.0 / 11.0 |
| W16-cache+EOS+trigger | window-16 Conf-0.9 + EOS truncation | 1.830 [1.702, 1.965] | 2.002 [1.862, 2.149] | 1.795 [1.666, 1.925] | 161.6 | 45.8 | 20/54 | +5 [-1, +11] (3/8) | +4 [-1, +10] (2/6) | 1.00 | 0.44 | 0.150 | 41.7 / 1.0 |
| W16-cache+EOS+trigger-keys | window-16 Conf-0.9 + EOS truncation | 1.772 [1.661, 1.899] | 1.938 [1.817, 2.076] | 1.738 [1.627, 1.855] | 158.4 | 48.3 | 16/54 | +1 [-3, +5] (2/3) | +0 [-5, +5] (3/3) | 1.00 | 0.61 | 0.202 | 56.0 / 1.0 |
| W16-cache+EOS+selective-r60 | window-16 Conf-0.9 + EOS truncation | 2.042 [1.906, 2.189] | 2.233 [2.085, 2.393] | 2.002 [1.864, 2.151] | 163.7 | 40.5 | 17/54 | +2 [-3, +8] (3/5) | +1 [-3, +5] (2/3) | 1.00 | 0.56 | 0.156 | 161.1 / 1.0 |
| W16-cache+EOS+selective-r80 | window-16 Conf-0.9 + EOS truncation | 1.963 [1.846, 2.091] | 2.147 [2.020, 2.287] | 1.925 [1.813, 2.043] | 153.6 | 44.9 | 16/54 | +1 [-4, +6] (3/4) | +0 [-5, +5] (3/3) | 1.00 | 0.44 | 0.246 | 151.8 / 1.0 |
| W16-cache+EOS+selective-r90 | window-16 Conf-0.9 + EOS truncation | 1.698 [1.599, 1.808] | 1.857 [1.748, 1.978] | 1.665 [1.568, 1.765] | 161.5 | 49.4 | 16/54 | +1 [-5, +7] (4/5) | +0 [-5, +6] (4/4) | 1.00 | 0.44 | 0.353 | 160.4 / 1.0 |

## Strata (indicative: 27 questions each)

| cell | stratum | speed vs Conf-0.9 | speed vs W16-dense | correct | net vs Conf-0.9 |
|---|---|---|---|---|---|
| SPARSE-none | L5 | 0.915 | 0.875 | 3/27 | +0 |
| SPARSE-none | L1-4 | 0.913 | 0.926 | 12/27 | +0 |
| W16-dense | L5 | 1.046 | - | 3/27 | +0 |
| W16-dense | L1-4 | 0.986 | - | 13/27 | +1 |
| W16-cache-block | L5 | 1.913 | 1.830 | 1/27 | -2 |
| W16-cache-block | L1-4 | 1.832 | 1.858 | 13/27 | +1 |
| W16-cache-block+EOS | L5 | 2.031 | 1.942 | 1/27 | -2 |
| W16-cache-block+EOS | L1-4 | 1.986 | 2.014 | 13/27 | +1 |
| W16-cache+EOS+trigger | L5 | 1.800 | 1.721 | 4/27 | +1 |
| W16-cache+EOS+trigger | L1-4 | 1.875 | 1.902 | 16/27 | +4 |
| W16-cache+EOS+trigger-keys | L5 | 1.807 | 1.728 | 2/27 | -1 |
| W16-cache+EOS+trigger-keys | L1-4 | 1.725 | 1.750 | 14/27 | +2 |
| W16-cache+EOS+selective-r60 | L5 | 2.058 | 1.968 | 2/27 | -1 |
| W16-cache+EOS+selective-r60 | L1-4 | 2.019 | 2.048 | 15/27 | +3 |
| W16-cache+EOS+selective-r80 | L5 | 2.048 | 1.958 | 3/27 | +0 |
| W16-cache+EOS+selective-r80 | L1-4 | 1.856 | 1.883 | 13/27 | +1 |
| W16-cache+EOS+selective-r90 | L5 | 1.741 | 1.664 | 2/27 | -1 |
| W16-cache+EOS+selective-r90 | L1-4 | 1.642 | 1.665 | 14/27 | +2 |

## Rows computed per layer, averaged over calls and questions (dense = prompt + answer rows; layers 0, 4, 8, 12, 16, 20, 24, 28, 31)

| cell | L0 | L4 | L8 | L12 | L16 | L20 | L24 | L28 | L31 | mean | calls with a refresh | rows joined per refresh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Conf-0.9 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | - | - |
| SPARSE-none | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | - | - |
| W16-dense | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 0.0 | 0 |
| W16-cache-block | 59 | 59 | 59 | 59 | 59 | 59 | 59 | 59 | 59 | 59 | 0.0 | 0 |
| W16-cache-block+EOS | 58 | 58 | 58 | 58 | 58 | 58 | 58 | 58 | 58 | 58 | 0.0 | 0 |
| W16-cache+EOS+trigger | 26 | 26 | 36 | 46 | 86 | 135 | 169 | 183 | 185 | 95 | 41.7 | 607 |
| W16-cache+EOS+trigger-keys | 26 | 30 | 52 | 78 | 126 | 180 | 227 | 239 | 239 | 129 | 56.0 | 606 |
| W16-cache+EOS+selective-r60 | 23 | 23 | 23 | 128 | 128 | 128 | 128 | 128 | 128 | 99 | 161.1 | 104 |
| W16-cache+EOS+selective-r80 | 23 | 23 | 23 | 219 | 219 | 219 | 219 | 219 | 219 | 164 | 151.8 | 195 |
| W16-cache+EOS+selective-r90 | 23 | 23 | 23 | 302 | 302 | 302 | 302 | 302 | 302 | 223 | 160.4 | 276 |

## Ladder: what each ingredient adds (speed ratio of summed seconds, calls ratio, net answers)

| step | speed | calls ratio | net answers [CI] |
|---|---|---|---|
| W16-dense vs Conf-0.9 (word lock change: window 16 + EOS truncation; no cache) | 1.020 [0.964, 1.096] | 0.986 | +1 [-3, +6] |
| W16-cache-block vs W16-dense (far-blank cache + block refresh) | 1.842 [1.693, 1.991] | 1.189 | -2 [-8, +3] |
| W16-cache-block+EOS vs W16-cache-block (EOS truncation) | 1.071 [1.028, 1.136] | 0.933 | +0 [+0, +0] |
| trigger vs block schedule (both with EOS truncation) | 0.910 [0.857, 0.968] | 0.930 | +6 [+0, +12] |
| trigger-keys vs trigger (cosine axis) | 0.968 [0.909, 1.038] | 0.980 | -4 [-9, +0] |
| selective-r60 vs trigger | 1.115 [1.042, 1.196] | 1.013 | -3 [-9, +3] |
| selective-r80 vs trigger | 1.072 [1.002, 1.152] | 0.951 | -4 [-10, +1] |
| selective-r90 vs trigger | 0.928 [0.862, 0.997] | 0.999 | -4 [-10, +2] |
| selective-r60 vs block schedule | 1.015 [0.967, 1.067] | 0.942 | +3 [-1, +8] |
| selective-r80 vs block schedule | 0.976 [0.920, 1.035] | 0.884 | +2 [-3, +7] |
| selective-r90 vs block schedule | 0.844 [0.807, 0.886] | 0.929 | +2 [-3, +7] |
| SPARSE-none vs Conf-0.9 (runtime overhead of row-by-row execution) | 0.914 [0.912, 0.917] | 1.000 | +0 [+0, +0] |

Peak GPU memory (GiB, max over questions): Conf-0.9 15.2, SPARSE-none 18.8, W16-dense 16.1, W16-cache-block 16.1, W16-cache-block+EOS 16.1, W16-cache+EOS+trigger 16.1, W16-cache+EOS+trigger-keys 16.1, W16-cache+EOS+selective-r60 16.2, W16-cache+EOS+selective-r80 16.2, W16-cache+EOS+selective-r90 16.2

