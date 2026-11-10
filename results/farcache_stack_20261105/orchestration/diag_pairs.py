import json, numpy as np, collections
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
by=collections.defaultdict(dict)
for l in open(OUT/'records/diag_he.jsonl'):
    r=json.loads(l); by[r['arm']][r['example_id']]=r
ids=sorted(by['D:C-0.9']); n=len(ids)
c={a:np.array([int(by[a][i]['correct']) for i in ids]) for a in by}
lat={a:np.array([by[a][i]['latency_s'] for i in ids]) for a in by}
rng=np.random.default_rng(23); idx=rng.integers(0,n,size=(10000,n))
out=[]
def pair(a,b):
    d=(c[a]-c[b]).astype(float)
    lo,hi=np.quantile(d[idx].sum(1),[.025,.975]); sp=lat[b].sum()/lat[a].sum(); slo,shi=np.quantile(lat[b][idx].sum(1)/lat[a][idx].sum(1),[.025,.975])
    return f"{a} vs {b}: answers {int(d.sum()):+d} [{lo:+.0f}, {hi:+.0f}], speed of {a} relative to {b} {sp:.3f} [{slo:.3f}, {shi:.3f}]"
for a,b in [('S32:block+k4','S32:main'),('S32:block+k2','S32:main'),('S32:nofar','S32:main'),('S32:block-noeos','S32:main'),('S16:block-noeos','S16:every+eos'),('S32:nofar','D:C-0.9'),('S32:block+k4','D:C-0.9'),('S32:block+k2','D:C-0.9'),('S32:block-noeos','D:C-0.9'),('S16:every+eos','D:C-0.9')]:
    out.append(pair(a,b)); print(out[-1])
# which tasks: are lost tasks of main those where the answer depends on a late-written part? text divergence position
def first_div(a,b):
    for k,(x,y) in enumerate(zip(a,b)):
        if x!=y: return k
    return None
lost=[i for i in ids if c['D:C-0.9'][ids.index(i)] and not c['S32:main'][ids.index(i)]]
fd=[first_div(by['S32:main'][i]['tokens'],by['D:C-0.9'][i]['tokens']) for i in lost]
allfd=[first_div(by['S32:main'][i]['tokens'],by['D:C-0.9'][i]['tokens']) for i in ids]
print('median first divergence position (tokens): lost',np.median([x for x in fd if x is not None]),' all',np.median([x for x in allfd if x is not None]), 'identical text all:',sum(x is None for x in allfd))
out.append(f"median first-divergence position of S32:main vs D:C-0.9: lost tasks {np.median([x for x in fd if x is not None]):.0f}, all tasks {np.median([x for x in allfd if x is not None]):.0f}; tasks with identical tokens {sum(x is None for x in allfd)}/{n}")
(OUT/'diag_he_pairs.md').write_text("\n".join("- "+o for o in out)+"\n")
