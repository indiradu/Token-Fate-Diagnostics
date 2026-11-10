import os, json, sys, ast, collections, re
rows=[json.loads(l) for l in open(sys.argv[1])]
by=collections.defaultdict(dict)
for r in rows:
    if r['set']=='he164' and r['status']=='ok': by[r['arm']][r['example_id']]=r
ids=sorted(set.intersection(*[set(v) for v in by.values()]))
base=by['D:C-0.9']
def code(t):
    m=re.search(r"```(?:python)?\n(.*?)(```|$)",t,re.S); return m.group(1) if m else t
def compiles(t):
    try: ast.parse(code(t)); return True
    except Exception: return False
print("syntax-valid share, all questions / lost questions")
for arm in sorted(by):
    lost=[i for i in ids if base[i]['correct'] and not by[arm][i]['correct']]
    print(f"{arm:14s} all {sum(compiles(by[arm][i]['text']) for i in ids)}/{len(ids)}  lost {sum(compiles(by[arm][i]['text']) for i in lost)}/{len(lost)}")
# tasks lost by many cache cells
cnt=collections.Counter()
for arm in by:
    if arm.startswith('S'):
        for i in ids:
            if base[i]['correct'] and not by[arm][i]['correct']: cnt[i]+=1
print("tasks lost in >=4 of 6 cache cells:", [(i,c) for i,c in cnt.most_common() if c>=4])
print("number of base-correct tasks:", sum(base[i]['correct'] for i in ids), " lost in >=1 cache cell:", len(cnt), " in all 6:", sum(c==6 for c in cnt.values()))
i=[k for k,c in cnt.most_common() if c>=1][0]
print("=== example", i); print("BASE:\n", base[i]['text'][:700]); print("S32:C-0.9:\n", by['S32:C-0.9'][i]['text'][:700])
