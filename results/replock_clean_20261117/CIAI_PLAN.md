# ciai smoke plan: does R0's derived schedule change on an A100, and does it matter? (PLAN ONLY, nothing run on ciai yet)

Written 2026-11-19. Owner decisions: partition `cscc-gpu-p`; project directory `/nfs-stor/rui.gao/Researches/phaselock`; 40 prompts per set for the comparison run; stay at the plan stage until told to go.

## Why
R0 derives its refresh schedule (rejoining rows, period k, refreshes per block) from two cost numbers measured on the GPU: F (fixed ms per call) and c (ms per row per layer), plus two universal constants (start layer floor(L/3), refresh budget 10% of block time). All results so far are from one RTX 5000 Ada. An offline derivation with scaled F and c changes the schedule on KodCode, GSM8K and MATH (e.g. GSM8K LLaDA-8B: all rows, k=4, 2 refreshes now; k=2, 5 refreshes if F doubles). The A100 is a second GPU with a different F/c ratio. Answer-free part: measure F, c and re-derive. Answer part: does the A100-derived schedule differ in speed or answers from the RTX schedule run on the same A100?

## What ciai is (read-only survey, 2026-11-19)
- Login `ciai-login-1`; ssh alias `ciai-copy` in ~/.ssh/config (key id_ed25519_ciai, user rui.gao). Plain `ssh ciai.mbzuai.ac.ae` does not use that key.
- Slurm, A100-SXM4-40GB, 4 per node. Partitions: `long` (default; needs `--qos=gpu-debug-qos`, max 3 h, 2 jobs per user, up to 4 GPUs), `cscc-gpu-p` (default QOS cscc-gpu-qos, no wall limit seen), `cscc-cpu-p`.
- Work area: `/nfs-stor/rui.gao/` (581 TB NFS, 93% full): existing `envs/`, `hf_cache/`, `uv_cache/`, `pip_cache/`, project folders (dof_dev). `/home` is 80% full and already holds 275 GB of ours: do not put envs or models there.
- Conventions seen in dof_dev: sbatch header `-p cscc-gpu-p --gres=gpu:1 -c 8 --mem=48G --time=..`, output to the project's `logs/%j.out`, python from `/nfs-stor/rui.gao/envs/<env>/bin/python`, `uv` for environments.
- No torch installed system-wide; `uv`, `rsync`, `git` present; outbound HTTPS works (huggingface.co reachable); no bwrap.
- Existing jobs of the owner there: `opichain-1`, `opichain-2` (2 GPUs each, pending on cscc-gpu-p, scheduler estimate to start 2026-11-20 23:51) and two interactive `bash` sessions on cscc-cpu-p running 2+ days. NOT TOUCHED. The file `HFtoken.md` in the interruptible workspace holds a token: never read or copy it.
- Dry-run admission (`sbatch --test-only`, nothing submitted): a new 1-GPU job on cscc-gpu-p is estimated to start about 2026-11-22 13:09 whatever the memory, CPUs or wall time (it queues behind opichain). On `long` with `--qos=gpu-debug-qos` the estimate was 2026-11-20 08:30.

## Consequence for the design
Every queued job waits about 2-3 days on cscc-gpu-p, so a ladder of separate jobs would take weeks. All GPU steps therefore go in ONE sbatch script run in one allocation (steps S3-S7 below), with early exit on the first failure and a log line per step. Steps S0-S2 need no GPU and run on the login node now. Before submitting, re-run `sbatch --test-only` on the real script.

## Ladder
- S0 (login, no GPU): `uv venv /nfs-stor/rui.gao/envs/phaselock`; install torch and the other packages pinned to the versions of our current venv (`pip freeze` of results/repfreeze_collect_20261006/venv_cu126), so software is identical across GPUs. Caches to `/nfs-stor/rui.gao/uv_cache`, `HF_HOME=/nfs-stor/rui.gao/hf_cache`.
- S1 (login): `rsync` into `/nfs-stor/rui.gao/Researches/phaselock/`: the orchestration runners (model_transfer_20261113/orchestration, replock_clean_20261117/orchestration), the frozen source snapshot (source_m1) and the prompt manifests. No weights, no records.
- S2 (login): download LLaDA-8B-Instruct (about 16 GB) to `/nfs-stor/rui.gao/models/` with `snapshot_download` (public repo, no token).
- S3 (in the GPU job, 10 min): `nvidia-smi`, torch sees the GPU, model loads, one dense forward timed. Fails fast if the stack is wrong.
- S4: cost benchmark (bench_layer_cost: dense call and query-only calls at 16..264 rows) gives F and c on the A100; derive R0 for the four sets and write a table next to the RTX 5000 Ada one. Answer-free, a few minutes.
- S5: smoke decode, 2 GSM8K prompts: dense Conf-0.9, plain cached (S32), R0 derived for the A100, R0 with the RTX constants. Check it runs, calls and timing are sane.
- S6: comparison run, 40 GSM8K and 40 HumanEval prompts, the same four arms interleaved prompt by prompt in one process (same GPU state). Speed against dense, calls, ms per call, paired answer change.
- S7: write records and a short report to `/nfs-stor/rui.gao/Researches/phaselock/records/`; copy back to our cluster for scoring (no bwrap on ciai, HumanEval is scored there on CPU) and analysis.

## Open details to settle before S3
- GPU-idle guard in our runners (`gpu_guard`, GPU_IN_USE markers) may misfire on ciai; use a bypass in a ciai-specific copy, never by editing the runners with pending jobs on the other cluster.
- Whether to add the controller arm (needs the 20-prompt recalibration; skip for the first run).
- If the opichain jobs start first, our job still waits behind them; the owner decides whether that matters.

## Pass criteria (set in advance, no tolerance imposed)
S3 passes if the model loads and a forward runs. S4 reports F and c with the derived schedules. S5 passes if all four arms finish with plausible calls. S6 reports, for each set: speed of the A100-derived R0 and of the RTX-constants R0 against dense on the A100, and the paired answer change with its interval. No pass/fail threshold on the comparison: it is a measurement.

## Progress 2026-11-19 (setup done, NO runs started)
- Owner clarified: no CPU cap on ciai (the env-build CPU job started at once; dry-run `--test-only` estimates are pessimistic there and must not be trusted for start times); 1 GPU is enough (LLaDA-8B bf16 about 15-18 GiB, A100 40 GB), and chaining as in the opichain jobs (short slices, 25-55 min) backfills into single free GPUs.
- Built: env `/nfs-stor/rui.gao/envs/phaselock` (Python 3.10.22, same pins as our cluster venv, torch 2.8.0+cu126; job 420011). Weights: `/nfs-stor/rui.gao/models/LLaDA-8B-Instruct` (15 GB, same revision 08b83a6 as our runs; every file of our local snapshot is present with identical byte size; ciai also has two small extra files, README.md and generation_config.json, which our manual decoding does not use; job 420028). Code and data in `/nfs-stor/rui.gao/Researches/phaselock/`: results/model_transfer_20261113/{orchestration,source_m1,original/manifests}, the RTX cost benchmark, and `ciai/` (run_transfer.py, sanity.py, slice.sbatch, submit_chain.sh, download_weights.sbatch, check_harness.sbatch); copies in this repo under `results/replock_clean_20261117/ciai/`.
- CPU harness check passed (job 420029): imports resolve with the frozen source, prompts load (GSM8K 100, HumanEval 164), RTX benchmark gives F 30.77 ms and c 0.0705, derived schedules equal our table.
- Chain design (staged, NOT submitted): `submit_chain.sh` = one slice for S3+S4+S5 (sanity, cost benchmark, 2-prompt smoke; afterok) then up to three S6 slices (comparison, afterany, resumable; each slice 55 min, GPU pre-flight, excludes gpu-05/07/51/52 as the opichain chain does). 1 GPU, 8 CPUs, 48 GB.
- Arms in the comparison: dense Conf-0.9, plain selective cache (S32), R0 derived from this GPU's benchmark, R0 derived from the RTX benchmark. 40 prompts of GSM8K and of HumanEval; HumanEval answers scored back on the RTX cluster.

## Amendment 2026-11-19: 30-minute slices, two comparison chains (owner)
Slices are now 30 min wall (about 25 min of work) instead of 55. `submit_chain.sh` (staged, not run): link 1 = S3+S4+S5 (sanity, this GPU's cost benchmark, 2-prompt smoke); then two independent chains, each up to two slices, one for GSM8K (S6G) and one for HumanEval (S6H), both waiting for link 1 (afterok, they need its cost benchmark). Slices inside a chain are linked with afterany and resume from the saved records (exit 75 when the work budget is used; a later slice exits at once if done). The two chains can use two free GPUs at once.

## Submission 2026-11-19 (owner: go)
Submitting the first chain hit `QOSMaxSubmitJobPerUserLimit`: about 4 submitted jobs per user on cscc-gpu-p, two of them the opichain jobs, so only 2 of mine were accepted (this is why the opichain chain has each slice submit its successor). Changed to the same pattern: `slice.sbatch` queues its own successor at start (`SUCC_DEPTH`), `submit_chain.sh` submits only link 1 and the first comparison slice. Submitted: pl-s345 = 420137 (S3 S4 S5), pl-cmp = 420143 (S6G S6H, afterok 420137, SUCC_DEPTH 3). The first attempt's accepted dependent job 420138 was cancelled and replaced. The scheduler's start estimates changed from Nov 22 to the same day (15:13 for 420137): estimates are unreliable.

## Results 2026-11-19 (chain finished in about 45 minutes; see ciai/results_transfer_a100.md)
- S3-S5 passed in 3 minutes. S4 benchmark: dense call 64 ms (RTX 81 ms); small query-only calls 33.3 ms (RTX 33.7 ms): the fixed per-call cost is nearly the same on both GPUs. Fit: F 28.6 ms, c 0.047 (RTX: 30.8, 0.0705).
- Derived R0: HumanEval identical on both GPUs (blank rows, k 2, 1 refresh); GSM8K differs (A100: all rows, k 3, 3 refreshes; RTX: all rows, k 4, 2 refreshes). A100-derived vs RTX-schedule R0 on the A100: speed 0.982x [0.944, 1.015] on GSM8K, 1.000x on HumanEval.
- Cache gain over dense on the A100: HumanEval 1.54x (RTX cluster 1.58x), GSM8K 1.08-1.10x (RTX cluster 1.36x): the gain shrinks where a dense call is close to the fixed per-call floor. 
- Answers (40 prompts, noisy): HumanEval R0 22/40 = dense 22/40, plain cached 20/40. GSM8K dense 22, S32 20, R0 A100-derived 18, R0 with the RTX schedule 29 (+7 [+2,+13] vs dense; +11 vs the A100-derived one). Not believed: on the RTX cluster the same 40 prompts gave dense 23 and R0 22, and outputs differ between GPUs on 26/40 (R0) and 18/40 (dense) prompts. A 100-prompt rerun is the check.
- Operational: slice start was immediate (no queue wait); the successor slices after completion ran the model load and exited (about 2 minutes each, wasteful: next time set SUCC_DEPTH lower or add a done-check before loading); submit limit about 4 jobs per user on cscc-gpu-p.
