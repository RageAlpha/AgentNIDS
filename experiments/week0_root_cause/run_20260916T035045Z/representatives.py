from pathlib import Path
import json,csv
from collections import Counter
import numpy as np
O=Path(__file__).resolve().parent;R=Path('/home/karri/projects/NIDSProject_Professor_Run');A=R/'outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z';F=R/'outputs/feature_store_v2';q=float.fromhex('0x1.51bb5a0000000p-22');forest=json.loads((O/'forest_structure.json').read_text());names=json.loads((F/'feature_store_manifest.json').read_text())['feature_order'];out=[];details=[]
for d in ['UNSW','ToN']:
 z=np.load(A/f'{d}_sample_features.npz');rr=z['ranks'];X=z['rf_input_float32'];yy=np.load(F/f'{d}_label.npy',mmap_mode='r')[rr];p=np.load(R/f'outputs/prediction_cache_v2/{d}/rf_All49.npy',mmap_mode='r')['raw_probability'][rr];sc=np.where(yy==0,p,1-p)
 if d=='UNSW':
  a=np.flatnonzero((rr<1780000)&(yy==0)&(p==q))[0];b=np.flatnonzero((rr<1780000)&(yy==1)&(sc>q));b=b[np.argmin(abs(p[b]-.9999995827674866))];chosen=[('source_Benign_at_q',a),('source_Attack_correct_but_rejected',b)]
 else:chosen=[('target_max_attack_probability',np.argmax(p))]
 for role,ix in chosen:
  probs=[];ttl=0;features=Counter()
  for ti,t in enumerate(forest['trees']):
   nodes={n['node_id']:n for n in t['nodes']};node=nodes[0];path=[]
   while 'leaf_value' not in node:
    f=node['split_feature_id'];v=X[ix,f];go=bool(v<=node['threshold']);path.append({'feature':names[f],'value':float(v),'threshold':node['threshold'],'left':go});features[names[f]]+=1;node=nodes[node['left_child'] if go else node['right_child']]
   lp=node['leaf_value'][1];probs.append(lp);ttl+=any(x['feature'] in ['MIN_TTL','MAX_TTL'] for x in path)
   if (role=='source_Benign_at_q' and lp>0) or (role=='target_max_attack_probability' and lp<1):details.append(dict(role=role,domain=d,rank=int(rr[ix]),tree=ti,leaf_attack_probability=lp,leaf_training_data_count=node.get('data_count'),path=json.dumps(path)))
  probs=np.asarray(probs);out.append(dict(role=role,domain=d,rank=int(rr[ix]),original_row=int(z['original_rows'][ix]),label=int(yy[ix]),cached_p=float(p[ix]),cached_hex=float(p[ix]).hex(),q_hex=q.hex(),reference_leaf_mean=float(probs.mean()),pure_Benign_trees=int((probs==0).sum()),pure_Attack_trees=int((probs==1).sum()),mixed_trees=int(((probs>0)&(probs<1)).sum()),TTL_path_trees=ttl,min_ttl=float(X[ix,24]),max_ttl=float(X[ix,25]),most_used_path_features=json.dumps(features.most_common(5))))
for n,rs in [('representative_tree_mechanisms.csv',out),('representative_leaf_paths.csv',details)]:
 with (O/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
print(json.dumps(out,indent=2))
