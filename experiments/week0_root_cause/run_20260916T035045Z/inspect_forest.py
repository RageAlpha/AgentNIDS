from pathlib import Path
import json,csv,pickle,hashlib
from collections import Counter
import numpy as np
O=Path(__file__).resolve().parent;R=Path('/home/karri/projects/NIDSProject_Professor_Run');A=R/'outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z'
def table(n,x):
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
features=json.loads((R/'outputs/feature_store_v2/feature_store_manifest.json').read_text())['feature_order'];modelpath=R/'outputs/source_v2/models/rf_All49/model.pkl'
with modelpath.open('rb') as f:m=pickle.load(f)
forest=json.loads(m.as_treelite().dump_as_json(pretty_print=False));(O/'forest_structure.json').write_text(json.dumps(forest,separators=(',',':')))
fi=m.feature_importances_;fi=fi.get() if hasattr(fi,'get') else np.asarray(fi)
root=Counter(t['nodes'][0].get('split_feature_id',-1) for t in forest['trees']);splits=Counter(n['split_feature_id'] for t in forest['trees'] for n in t['nodes'] if 'split_feature_id' in n)
table('frozen_model_feature_importance.csv',[dict(feature=features[i],importance=float(fi[i]),root_trees=root[i],split_nodes=splits[i],interpretation='training impurity importance; not causal attribution') for i in np.argsort(-fi)])
rows=[];allmaxdiff=0
for d in ['UNSW','ToN']:
 z=np.load(A/f'{d}_sample_features.npz');X=z['rf_input_float32'];ranks=z['ranks'];n=len(X);yy=np.load(R/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')[ranks];cached=np.load(R/f'outputs/prediction_cache_v2/{d}/rf_All49.npy',mmap_mode='r')['raw_probability'][ranks]
 sums=np.zeros(n);pure0=np.zeros(n,dtype=int);pure1=pure0.copy();mixed=pure0.copy();pathTTL=pure0.copy();leaveslow=pure0.copy();hist=Counter()
 for ti,t in enumerate(forest['trees']):
  nodes=t['nodes'];size=max(x['node_id'] for x in nodes)+1;feature=np.full(size,-1,dtype=int);thr=np.zeros(size,np.float32);left=np.zeros(size,dtype=int);right=left.copy();leaf=np.zeros(size);default=np.zeros(size,bool)
  for node in nodes:
   i=node['node_id']
   if 'leaf_value' in node:leaf[i]=node['leaf_value'][1]
   else:
    assert node['comparison_op']=='<=';feature[i]=node['split_feature_id'];thr[i]=node['threshold'];left[i]=node['left_child'];right[i]=node['right_child'];default[i]=node['default_left']
  at=np.zeros(n,dtype=int);seen=np.zeros(n,bool)
  while True:
   ix=np.flatnonzero(feature[at]>=0)
   if not len(ix):break
   nd=at[ix];ff=feature[nd];seen[ix]|=np.isin(ff,[24,25]);v=X[ix,ff];go=np.where(np.isnan(v),default[nd],v<=thr[nd]);at[ix]=np.where(go,left[nd],right[nd])
  p=leaf[at];sums+=p;pure0+=p==0;pure1+=p==1;mixed+=(p>0)&(p<1);leaveslow+=p<.5;pathTTL+=seen
 avg=sums/len(forest['trees']);diff=np.abs(avg-cached);allmaxdiff=max(allmaxdiff,float(diff.max()))
 partition=np.where(ranks<1780000,'S_prob_cal','S_source_test') if d=='UNSW' else np.full(n,'T_heldout')
 for part in np.unique(partition):
  for y in [0,1]:
   mask=(partition==part)&(yy==y)
   for quantity,values in [('mean_tree_attack_probability',avg),('pure_Benign_leaf_trees',pure0),('pure_Attack_leaf_trees',pure1),('mixed_leaf_trees',mixed),('trees_with_attack_leaf_p_below_half',leaveslow),('trees_using_TTL_on_path',pathTTL),('reference_mean_minus_cached_abs',diff)]:
    vals=values[mask];quant=np.quantile(vals,[0,.1,.5,.9,1])
    rows.append(dict(domain=d,partition=part,y=y,quantity=quantity,n=len(vals),min=float(quant[0]),p10=float(quant[1]),median=float(quant[2]),p90=float(quant[3]),max=float(quant[4]),scope='existing 53813 directed audit samples; descriptive mechanism trace, not performance estimate'))
 print('TRAVERSED',d,'max mean-vs-cache error',diff.max(),flush=True)
table('sampled_tree_path_diagnostics.csv',rows)
(O/'forest_inspection_validation.json').write_text(json.dumps({'status':'COMPLETED_DIAGNOSTIC','same_model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'trees':len(forest['trees']),'average_tree_output':forest['average_tree_output'],'postprocessor':forest['postprocessor'],'max_reference_float64_leaf_mean_vs_native_float32_abs_error':allmaxdiff,'reference_not_used_for_pilot_predictions':True,'no_model_mutation_or_retraining':True,'important_limit':'Tree export traversed read-only for mechanism. Float64 leaf summation is not native GPU accumulation and is not a bit equality claim.'},indent=2)+'\n')
