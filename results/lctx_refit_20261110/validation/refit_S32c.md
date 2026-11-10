# Refit Lctx-0.8 on S32c one-word-per-call traces

Rows (non-EOS/EOT proposals): fit 481410, threshold 159576; label mean fit 0.669, threshold 0.644.

| quantity (threshold-question rows) | refit (primary) | refit, original row filter | original rule as is | original rule, lambda recalibrated here | C-0.8 |
|---|---|---|---|---|---|
| lambda | 0.9788 | 0.9784 | 0.9605 | 0.9651 | 0.8 (confidence) |
| admitted share | 0.349 | 0.357 | 0.353 | 0.346 | 0.337 |
| pooled precision of the admitted rows | 0.9973 | 0.9973 | 0.9969 | (target) | 0.9973 |
| AUROC (label = proposal equals final token) | 0.9271 | 0.9274 | 0.9270 | same | confidence alone 0.9250 |

AUROC refit minus original, cluster bootstrap over the 20 threshold questions: +0.0001 [-0.0002085378159259077, 0.00047827585770631944].

Feature ranges, new fit rows vs the original fit rows (min / 0.1% / median / 99.9% / max) and the share of new rows outside the original [min, max]:

| feature | original | new | share outside original range |
|---|---|---|---|
| confidence | 0.0335 / 0.0478 / 0.603 / 1 / 1 | 0.0313 / 0.0455 / 0.54 / 1 / 1 | 0.0000 |
| entropy | 3.54e-07 / 2.31e-05 / 1.38 / 5.12 / 5.69 | 1.48e-07 / 4.92e-05 / 1.62 / 5.08 / 5.55 | 0.0000 |
| margin | 0 / 0 / 0.423 / 1 / 1 | 0 / 0 / 0.34 / 1 / 1 | 0.0000 |
| dist_written | 1 / 1 / 8 / 31 / 31 | 1 / 1 / 8 / 512 / 512 | 0.0057 |
| local_mask_ratio | 0.111 / 0.111 / 1 / 1 / 1 | 0.111 / 0.111 / 1 / 1 / 1 | 0.0000 |
| pos_in_block | 0.0312 / 0.0312 / 0.312 / 0.969 / 0.969 | 0.0312 / 0.0312 / 0.281 / 0.969 / 0.969 | 0.0000 |
