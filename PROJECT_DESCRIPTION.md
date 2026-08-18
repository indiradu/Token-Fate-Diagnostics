# Project Description

Token-Fate Diagnostics is a Python research repository for studying premature
token commitment in diffusion language models. It treats token fate as the
question of whether an intermediate denoising token agrees with the final
decoded token, then evaluates trace features, hidden-state probes, and strict
counterfactual interventions on LLaDA-style confidence decoding.

The public repo contains source code under `src/regret_remasking/`, experiment
and analysis entrypoints under `scripts/`, lightweight tests under `tests/`,
protocol/result notes under `docs/`, and compact aggregate reported artifacts
under `results/`. Raw traces, hidden-state arrays, local model files, and
scheduler outputs are intentionally excluded.

The current published interpretation is that token fate is strongly predictable
from traces, but confidence explains much of the broad trace-regret signal. The
clearest positive held-out result is a frozen general trace predictor on
high-confidence GSM8K tokens, while strict confidence- and trajectory-matched
interventions do not establish a resolved causal enrichment beyond confidence.

Local verification that does not load LLaDA:

```bash
python3 -m compileall src scripts tests
python3 tests/test_features.py
python3 tests/test_counterfactual_selection.py
```
