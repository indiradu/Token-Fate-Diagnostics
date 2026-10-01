from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from regret_remasking.paper_collection import collect_hidden_manifest
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--projector-checkpoint',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=23);p.add_argument('--storage-ledger');p.add_argument('--max-storage-gib',type=float,default=600)
a=p.parse_args();r=collect_hidden_manifest(a.manifest,a.output_dir,a.checkpoint,a.projector_checkpoint,device=a.device,seed=a.seed,storage_ledger=a.storage_ledger,max_storage_gib=a.max_storage_gib)
print(json.dumps({k:v for k,v in r.items() if k!='examples'},indent=2))
