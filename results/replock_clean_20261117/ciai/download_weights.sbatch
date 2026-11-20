#!/bin/bash
#SBATCH -J pl-weights
#SBATCH -p cscc-cpu-p
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /nfs-stor/rui.gao/Researches/phaselock/logs/weights-%j.out
# CPU-only: download LLaDA-8B-Instruct at the SAME revision as our RTX cluster runs (08b83a6...) to /nfs-stor/rui.gao/models. Public repo, no token. Caches on /nfs-stor.
set -e
export HF_HOME=/nfs-stor/rui.gao/hf_cache HF_HUB_DISABLE_XET=1 TMPDIR=/nfs-stor/rui.gao/tmp
mkdir -p $TMPDIR
/nfs-stor/rui.gao/envs/phaselock/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("GSAI-ML/LLaDA-8B-Instruct", revision="08b83a6feb34df1a6011b80c3c00c7563e963b07", local_dir="/nfs-stor/rui.gao/models/LLaDA-8B-Instruct")
print("done", p)
PY
du -sh /nfs-stor/rui.gao/models/LLaDA-8B-Instruct; ls -la /nfs-stor/rui.gao/models/LLaDA-8B-Instruct | head -30
