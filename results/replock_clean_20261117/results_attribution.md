# Where the damage starts (descriptive)
For tasks the same stack's dense Conf-0.9 answers correctly: first token that differs from the dense decode. 'lost' = the arm answers wrongly; 'diverged, not lost' = the decode differs but the answer status is unchanged.
Offset in block 0-31: a divergence at a low offset right after a block start means the first calls of a block already differ; a high offset means staleness built up during the block.

| arm | lost tasks | median first-diff token (lost) | median block (lost) | share in the first 2 blocks (lost) | median offset in block (lost) | share offset >= 16 (lost) | diverged-not-lost: median first-diff token | median block |
|---|---|---|---|---|---|---|---|---|
| R0 | 197 | 35 | 1 | 61% | 15 | 48% | 32 | 1 |
| CTL | 193 | 25 | 0 | 73% | 14 | 45% | 27 | 0 |
| ELASTIC | 90 | 30 | 0 | 66% | 16 | 50% | 38 | 1 |
| FD_DUAL | 145 | 25 | 0 | 72% | 16 | 52% | 23 | 0 |
| DLLM | 149 | 21 | 0 | 72% | 10 | 24% | 26 | 0 |
| SURELOCK | 82 | 70 | 2 | 45% | 16 | 52% | 78 | 2 |