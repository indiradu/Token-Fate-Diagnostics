# Feature ablation 3 (provenance h, structure i, neighbour window k, call-relative j; lookahead reference O)

G1 traces, rows not EOS/EOT and not written on its own call: fit 458265 (60 questions), threshold 148386 (20 questions). C-0.8 pooled precision 0.99514, share 0.3464; raw-confidence AUROC 0.9268. Cluster bootstrap over threshold questions, 2000 resamples, target and lambda re-chosen per resample; differences paired. Share = admitted share at C-0.8's pooled precision.

| set | cols | AUROC | share | precision | AUROC diff vs six | share diff vs six, points | share diff vs six+b, points | share diff vs C-0.8, points |
|---|---|---|---|---|---|---|---|---|
| conf | 1 | 0.9267 | 0.3452 | 0.9952 | -0.0026 [-0.0034, -0.0015] | -1.55 [-2.21, -0.89] | -2.56 [-3.19, -1.67] | -0.12 [-0.46, -0.10] |
| six | 6 | 0.9293 | 0.3607 | 0.9951 | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | -1.01 [-1.68, -0.28] | +1.43 [+0.77, +2.04] |
| six+b | 12 | 0.9325 | 0.3708 | 0.9951 | +0.0032 [+0.0023, +0.0041] | +1.01 [+0.28, +1.68] | +0.00 [+0.00, +0.00] | +2.44 [+1.55, +3.02] |
| six+h | 11 | 0.9293 | 0.3602 | 0.9951 | +0.0000 [-0.0003, +0.0004] | -0.04 [-0.33, +0.45] | -1.05 [-1.75, -0.26] | +1.39 [+0.68, +2.29] |
| six+i | 11 | 0.9291 | 0.3624 | 0.9951 | -0.0002 [-0.0004, +0.0001] | +0.18 [-0.33, +0.44] | -0.84 [-1.68, -0.11] | +1.60 [+0.80, +2.05] |
| six+k | 10 | 0.9303 | 0.3699 | 0.9952 | +0.0010 [+0.0004, +0.0017] | +0.92 [-0.08, +1.99] | -0.09 [-1.03, +1.15] | +2.35 [+1.47, +3.29] |
| six+j | 9 | 0.9298 | 0.3675 | 0.9951 | +0.0005 [+0.0001, +0.0010] | +0.69 [-0.15, +1.45] | -0.32 [-1.01, +0.46] | +2.12 [+1.22, +2.72] |
| six+hikj | 23 | 0.9310 | 0.3730 | 0.9951 | +0.0017 [+0.0011, +0.0023] | +1.23 [+0.67, +1.97] | +0.22 [-0.37, +1.16] | +2.66 [+2.11, +3.40] |
| six+b+k | 16 | 0.9322 | 0.3722 | 0.9951 | +0.0029 [+0.0022, +0.0037] | +1.16 [+0.44, +1.83] | +0.15 [-0.32, +0.78] | +2.59 [+1.65, +3.25] |
| six+b+hikj | 29 | 0.9321 | 0.3692 | 0.9951 | +0.0028 [+0.0019, +0.0036] | +0.85 [+0.16, +1.68] | -0.16 [-0.70, +0.60] | +2.28 [+1.58, +2.97] |
| six+O | 10 | 0.9522 | 0.4645 | 0.9951 | +0.0229 [+0.0189, +0.0272] | +10.39 [-34.93, +12.57] | +9.37 [-35.89, +11.68] | +11.81 [-33.96, +14.25] |
| six+b+hikj+O | 33 | 0.9650 | 0.5009 | 0.9951 | +0.0357 [+0.0312, +0.0403] | +14.03 [+10.76, +16.25] | +13.02 [+9.73, +15.30] | +15.45 [+11.96, +17.85] |
| six+noise17 | 23 | 0.9284 | 0.3609 | 0.9951 | -0.0009 [-0.0013, -0.0006] | +0.03 [-0.38, +0.23] | -0.99 [-1.73, -0.37] | +1.45 [+0.56, +2.00] |
| six+hikj_shuf | 23 | 0.9287 | 0.3612 | 0.9951 | -0.0006 [-0.0009, -0.0003] | +0.05 [-0.14, +0.29] | -0.96 [-1.63, -0.18] | +1.48 [+0.85, +2.05] |

Cost of groups h/i/k/j: 0.145 ms per call (mean over questions; max 0.162).

NaN share on threshold rows: h_bigram_prompt 0.911, h_bigram_written 0.911, i_after_eq 0.797, i_after_op 0.797, k_min_w2 0.019, k_mean_w4 0.010, o_nbr_y_l 0.039, o_nbr_y_r 0.059

## Permutation importance (AUROC drop, 3 repeats)

**six+hikj** (base 0.9310): group six 0.31580, margin 0.14167, confidence 0.06167, group k 0.02955, k_mean_w4 0.01917, entropy 0.00836, k_min_w2 0.00822, group j 0.00205, k_run 0.00195, j_conf_minus_med 0.00135, group i 0.00063, local_mask_ratio 0.00056, j_rank_frac 0.00049, group h 0.00046

**six+b+hikj** (base 0.9321): group six 0.35871, margin 0.18911, confidence 0.17080, entropy 0.13128, group b 0.00995, group k 0.00790, k_mean_w4 0.00321, same_l 0.00317, k_min_w2 0.00302, nbr_conf_l 0.00282, nbr_conf_r 0.00160, same_r 0.00149, group j 0.00102, k_run 0.00096

**six+O** (base 0.9522): group O 0.33001, o_conf_next 0.21322, margin 0.15176, group six 0.14944, confidence 0.14313, entropy 0.11308, o_same_next 0.02667, o_nbr_y_l 0.01269, o_nbr_y_r 0.00978, local_mask_ratio 0.00077, pos_in_block 0.00056, dist_written -0.00038

## G2 shift (C-0.8 multi-write, 80 questions, 174157 rows not written on their call, y mean 0.427)

| feature | G1 fit p10 / p50 / p90 | G2 p10 / p50 / p90 | G2 share outside G1 fit range |
|---|---|---|---|
| h_in_prompt | 0 / 1 / 1 | 0 / 1 / 1 | 0.000 |
| h_in_written | 0 / 1 / 1 | 0 / 1 / 1 | 0.000 |
| h_bigram_prompt | 0 / 0 / 1 | 0 / 0 / 1 | 0.000 |
| h_bigram_written | 0 / 1 / 1 | 0 / 0 / 1 | 0.000 |
| h_num_novel | 0 / 0 / 0 | 0 / 0 / 0 | 0.000 |
| i_digit | 0 / 0 / 1 | 0 / 0 / 1 | 0.000 |
| i_after_eq | 0 / 0 / 0 | 0 / 0 / 0 | 0.000 |
| i_after_op | 0 / 0 / 0 | 0 / 0 / 0 | 0.000 |
| i_in_math | 0 / 0 / 0 | 0 / 0 / 0 | 0.000 |
| i_brace_depth | 0 / 0 / 1 | 0 / 0 / 0 | 0.000 |
| k_min_w2 | 0.0875 / 0.349 / 0.989 | 0.0703 / 0.149 / 0.483 | 0.000 |
| k_mean_w4 | 0.117 / 0.528 / 0.989 | 0.0869 / 0.243 / 0.606 | 0.000 |
| k_run | 0 / 1 / 407 | 0 / 0 / 54 | 0.000 |
| k_written_w4 | 0 / 0 / 3 | 0 / 0 / 2 | 0.000 |
| j_rank_frac | 0.125 / 0.5 / 0.875 | 0.111 / 0.462 / 0.844 | 0.000 |
| j_conf_minus_med | -0.229 / -0.000588 / 0.285 | -0.177 / -0.00729 / 0.21 | 0.000 |
| j_block_written | 0.0312 / 0.281 / 0.656 | 0 / 0.219 / 0.656 | 0.000 |

AUROC of the G1-fitted models on G2 not-written rows: conf 0.8361, six 0.8406, six+b 0.8474, six+h 0.8410, six+i 0.8406, six+k 0.8435, six+j 0.8423, six+hikj 0.8440, six+b+hikj 0.8478

## Validity

```
{
 "V1_questions": 80,
 "V1_failed": [],
 "V1_ok": true,
 "V2": {
  "auroc": 0.9353917110126422,
  "share": 0.3951360660512353,
  "ref_auroc": 0.9353917110126422,
  "ref_share": 0.3951360660512353,
  "abs_diff": [
   0.0,
   0.0
  ],
  "ok": true
 },
 "V2_ok": true,
 "V3_violations": {
  "k_min_w2_above_nbr_conf_l": 0,
  "k_min_w2_above_nbr_conf_r": 0,
  "k_run_vs_conf": 0
 },
 "V3_ok": true,
 "V4_range_violations": {},
 "V4_prompt_length_mismatch": [],
 "V4_ok": true
}
```

## Gate

```
{
 "best_set": "six+hikj",
 "best_share_diff_vs_six": {
  "point": 0.012319221489898002,
  "mean": 0.01271071608502638,
  "ci95": [
   0.00672420389996405,
   0.019683592999914342
  ],
  "p_gt0": 1.0
 },
 "best_auroc_diff_vs_six": {
  "point": 0.0017031068189956589,
  "mean": 0.001704420131563632,
  "ci95": [
   0.001115484847450296,
   0.0022724548017775234
  ],
  "p_gt0": 1.0
 },
 "controls_share_diff_vs_six": {
  "six+noise17": 0.00026956720984461224,
  "six+hikj_shuf": 0.0005458735999353537
 },
 "conditions": {
  "G0_validity": true,
  "G1_share": false,
  "G2_auroc": false,
  "G3_controls": true,
  "G4_cost": true
 },
 "passes": false,
 "reference_six+O_share_diff": {
  "point": 0.1038575067728762,
  "mean": 0.07196074850846182,
  "ci95": [
   -0.34926776737667925,
   0.1256932781970974
  ],
  "p_gt0": 0.9365
 },
 "reference_reaches_gate_size": true
}
```
