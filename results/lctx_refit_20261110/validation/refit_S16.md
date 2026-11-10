# Refit Lctx-0.8 on S16 one-word-per-call traces

Rows (non-EOS/EOT proposals): fit 455222, threshold 157915; label mean fit 0.704, threshold 0.691.

| quantity (threshold-question rows) | refit (primary) | refit, original row filter | original rule as is | original rule, lambda recalibrated here | C-0.8 |
|---|---|---|---|---|---|
| lambda | 0.9754 | 0.9742 | 0.9605 | 0.9991 | 0.8 (confidence) |
| admitted share | 0.350 | 0.356 | 0.470 | 0.233 | 0.336 |
| pooled precision of the admitted rows | 0.9927 | 0.9927 | 0.9237 | (target) | 0.9927 |
| AUROC (label = proposal equals final token) | 0.9071 | 0.9075 | 0.8651 | same | confidence alone 0.9065 |

AUROC refit minus original, cluster bootstrap over the 20 threshold questions: +0.0416 [0.028693013291102458, 0.05426087770202773].

Feature ranges, new fit rows vs the original fit rows (min / 0.1% / median / 99.9% / max) and the share of new rows outside the original [min, max]:

| feature | original | new | share outside original range |
|---|---|---|---|
| confidence | 0.0335 / 0.0478 / 0.603 / 1 / 1 | 0.036 / 0.0487 / 0.572 / 1 / 1 | 0.0000 |
| entropy | 3.54e-07 / 2.31e-05 / 1.38 / 5.12 / 5.69 | 5.29e-07 / 5.17e-05 / 1.42 / 5.23 / 5.5 | 0.0000 |
| margin | 0 / 0 / 0.423 / 1 / 1 | 0 / 0 / 0.368 / 1 / 1 | 0.0000 |
| dist_written | 1 / 1 / 8 / 31 / 31 | 1 / 1 / 6 / 512 / 512 | 0.0021 |
| local_mask_ratio | 0.111 / 0.111 / 1 / 1 / 1 | 0.111 / 0.111 / 1 / 1 / 1 | 0.0000 |
| pos_in_block | 0.0312 / 0.0312 / 0.312 / 0.969 / 0.969 | 0.0312 / 0.0312 / 0.75 / 0.969 / 0.969 | 0.0000 |
