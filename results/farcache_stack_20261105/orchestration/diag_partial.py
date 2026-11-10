import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import json, sys, collections
from pathlib import Path
rows=[json.loads(l) for l in open(sys.argv[1])]
by=collections.defaultdict(dict)
for r in rows:
    if r['set']=='he164' and r['status']=='ok': by[r['arm']][r['example_id']]=r
ids=sorted(set.intersection(*[set(v) for v in by.values()]))
print(len(ids),'questions')
EOS,EOT=126081,126348
def feats(r):
    t=r['tokens']; fe=r['first_eos']
    body=t[:fe] if fe is not None else t
    # repetition: longest run of one token; share of 4-gram repeats
    run=1;best=1
    for a,b in zip(body,body[1:]):
        run=run+1 if a==b else 1; best=max(best,run)
    g=[tuple(body[i:i+4]) for i in range(max(len(body)-3,0))]
    rep=1-len(set(g))/len(g) if g else 0
    return dict(first_eos=fe,len=len(body),maxrun=best,rep4=rep,status=r['scoring_status'])
base=by['D:C-0.9']
for arm in sorted(by):
    if arm.startswith('D:') and arm=='D:C-0.9': continue
    lost=[i for i in ids if base[i]['correct'] and not by[arm][i]['correct']]
    gained=[i for i in ids if not base[i]['correct'] and by[arm][i]['correct']]
    st=collections.Counter(by[arm][i]['scoring_status'] for i in lost)
    stb=collections.Counter(base[i]['scoring_status'] for i in ids if not base[i]['correct'])
    L=[feats(by[arm][i]) for i in lost]
    allf=[feats(by[arm][i]) for i in ids]; bf=[feats(base[i]) for i in ids]
    import statistics as S
    med=lambda xs,k: S.median([x[k] for x in xs if x[k] is not None]) if xs else None
    print(f"{arm:14s} lost {len(lost):2d} gained {len(gained):2d} | lost status {dict(st)} | lost: median len {med(L,'len')} maxrun {med(L,'maxrun')} rep4 {med(L,'rep4'):.2f} | all arm: len {med(allf,'len')} rep4 {med(allf,'rep4'):.2f} maxrun>=8 {sum(f['maxrun']>=8 for f in allf)} | base len {med(bf,'len')} rep4 {med(bf,'rep4'):.2f}")
    print("     first-EOS earlier than base by >=20 tokens in lost:", sum(1 for i in lost if (by[arm][i]['first_eos'] or 512) < (base[i]['first_eos'] or 512)-20), "of", len(lost), "| EOS later:", sum(1 for i in lost if (by[arm][i]['first_eos'] or 512) > (base[i]['first_eos'] or 512)+20))
print("base wrong statuses:",dict(stb))
