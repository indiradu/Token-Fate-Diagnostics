# Pooled paired answer change (Study 5)
Change in correct answers as a share of prompts (percentage points), pooled over cells; 95% bootstrap over prompts within each cell. Reference (a): the model's original decoder; (b): the arm's own stack's dense Conf-0.9 decode.

## (a) against the original decoder
| arm | all cells | excluding LLaDA-MoE | LLaDA-8B | LLaDA-1.5 | Dream | LLaDA-MoE |
|---|---|---|---|---|---|---|
| R0 (ours) | -2.7 pts [-4.5, -0.7]  n=1980, 15 cells | -1.9 pts [-4.0, +0.3] (incl. 0)  n=1458, 11 cells | -0.8 pts [-4.6, +2.9] (incl. 0)  n=522, 4 cells | -2.1 pts [-6.1, +1.7] (incl. 0)  n=522, 4 cells | -2.9 pts [-7.0, +1.2] (incl. 0)  n=414, 3 cells | -5.0 pts [-8.4, -1.5]  n=522, 4 cells |
| Controller (ours) | -2.9 pts [-4.7, -1.0]  n=1980, 15 cells | -1.9 pts [-4.1, +0.3] (incl. 0)  n=1458, 11 cells | -2.5 pts [-6.1, +1.1] (incl. 0)  n=522, 4 cells | -2.9 pts [-6.7, +1.0] (incl. 0)  n=522, 4 cells | +0.0 pts [-3.9, +4.1] (incl. 0)  n=414, 3 cells | -5.6 pts [-9.2, -1.9]  n=522, 4 cells |
| Elastic-Cache | -0.5 pts [-3.0, +2.0] (incl. 0)  n=1044, 8 cells | -0.5 pts [-3.0, +2.0] (incl. 0)  n=1044, 8 cells | +1.1 pts [-1.9, +4.2] (incl. 0)  n=522, 4 cells | -2.1 pts [-5.7, +1.7] (incl. 0)  n=522, 4 cells | – | – |
| Fast-dLLM dual | -2.3 pts [-4.5, -0.1]  n=1458, 11 cells | -2.3 pts [-4.6, -0.1]  n=1458, 11 cells | -1.1 pts [-4.8, +2.5] (incl. 0)  n=522, 4 cells | -3.3 pts [-6.9, +0.8] (incl. 0)  n=522, 4 cells | -2.7 pts [-7.2, +1.4] (incl. 0)  n=414, 3 cells | – |
| dLLM-Cache | -3.2 pts [-5.3, -1.0]  n=1458, 11 cells | -3.2 pts [-5.3, -1.0]  n=1458, 11 cells | -2.1 pts [-5.6, +1.3] (incl. 0)  n=522, 4 cells | -2.1 pts [-5.7, +1.7] (incl. 0)  n=522, 4 cells | -5.8 pts [-9.7, -1.9]  n=414, 3 cells | – |
| SureLock | -1.7 pts [-4.1, +0.6] (incl. 0)  n=1044, 8 cells | -1.7 pts [-4.0, +0.6] (incl. 0)  n=1044, 8 cells | -0.8 pts [-4.0, +2.3] (incl. 0)  n=522, 4 cells | -2.7 pts [-6.1, +0.6] (incl. 0)  n=522, 4 cells | – | – |

## (b) against the same stack's dense Conf-0.9
| arm | all cells | excluding LLaDA-MoE | LLaDA-8B | LLaDA-1.5 | Dream | LLaDA-MoE |
|---|---|---|---|---|---|---|
| R0 (ours) | -1.5 pts [-3.4, +0.4] (incl. 0)  n=1980, 15 cells | -0.6 pts [-2.8, +1.6] (incl. 0)  n=1458, 11 cells | -0.4 pts [-4.0, +3.3] (incl. 0)  n=522, 4 cells | -0.2 pts [-4.2, +3.6] (incl. 0)  n=522, 4 cells | -1.4 pts [-5.3, +2.4] (incl. 0)  n=414, 3 cells | -3.8 pts [-7.3, -0.4]  n=522, 4 cells |
| Controller (ours) | -1.7 pts [-3.5, +0.2] (incl. 0)  n=1980, 15 cells | -0.7 pts [-2.9, +1.4] (incl. 0)  n=1458, 11 cells | -2.1 pts [-5.7, +1.5] (incl. 0)  n=522, 4 cells | -1.0 pts [-4.6, +2.7] (incl. 0)  n=522, 4 cells | +1.4 pts [-2.4, +5.3] (incl. 0)  n=414, 3 cells | -4.4 pts [-8.0, -0.8]  n=522, 4 cells |
| Elastic-Cache | +0.7 pts [-2.0, +3.3] (incl. 0)  n=1044, 8 cells | +0.7 pts [-2.0, +3.2] (incl. 0)  n=1044, 8 cells | +1.5 pts [-1.7, +5.0] (incl. 0)  n=522, 4 cells | -0.2 pts [-4.0, +3.8] (incl. 0)  n=522, 4 cells | – | – |
| Fast-dLLM dual | -1.3 pts [-3.5, +1.0] (incl. 0)  n=1458, 11 cells | -1.3 pts [-3.6, +1.0] (incl. 0)  n=1458, 11 cells | -0.6 pts [-4.2, +2.9] (incl. 0)  n=522, 4 cells | -2.3 pts [-6.1, +1.2] (incl. 0)  n=522, 4 cells | -1.0 pts [-5.1, +3.1] (incl. 0)  n=414, 3 cells | – |
| dLLM-Cache | -2.1 pts [-4.4, +0.1] (incl. 0)  n=1458, 11 cells | -2.1 pts [-4.5, +0.1] (incl. 0)  n=1458, 11 cells | -1.5 pts [-5.4, +2.3] (incl. 0)  n=522, 4 cells | -1.1 pts [-4.6, +2.3] (incl. 0)  n=522, 4 cells | -4.1 pts [-8.0, -0.2]  n=414, 3 cells | – |
| SureLock | -0.6 pts [-3.0, +1.8] (incl. 0)  n=1044, 8 cells | -0.6 pts [-2.9, +1.8] (incl. 0)  n=1044, 8 cells | -0.4 pts [-3.6, +2.9] (incl. 0)  n=522, 4 cells | -0.8 pts [-4.2, +2.7] (incl. 0)  n=522, 4 cells | – | – |
