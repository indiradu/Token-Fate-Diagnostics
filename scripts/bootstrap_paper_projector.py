from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from regret_remasking.paper_collection import bootstrap_projector
p=argparse.ArgumentParser();p.add_argument('--fit-manifest',required=True);p.add_argument('--bootstrap-ids',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=1000);p.add_argument('--samples-per-example',type=int,default=512)
a=p.parse_args();r=bootstrap_projector(a.fit_manifest,a.bootstrap_ids,a.output_dir,a.checkpoint,device=a.device,seed=a.seed,fit_steps=a.steps,samples_per_example=a.samples_per_example)
print(json.dumps({k:v for k,v in r.items() if k!='diagnostics'},indent=2))
