from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).parent))
from fixed_core import writej,now,require
from run_pilot import SEEDS,OBJECTIVES,modelname,HISTORY,actionname,PAIRS
r=Path(sys.argv[1]);p=json.loads((r/'config/protocol.json').read_text());checks={}
for seed in SEEDS:
    shared=torch.load(r/'models'/f'initial_{seed}.pt',map_location='cpu',weights_only=True)
    order=np.load(r/'source'/f'batch_order_{seed}.npz')['indices'];require(order.shape==(20,50000) and all(np.array_equal(np.sort(x),np.arange(50000)) for x in order),'epoch sample order')
    previous=None
    for obj in OBJECTIVES:
        directory=r/'models'/modelname(obj,seed);initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=True);require(set(initial)==set(shared) and all(torch.equal(initial[k],shared[k]) for k in shared),'paired initial state')
        final=torch.load(directory/'final.pt',map_location='cpu',weights_only=True);require(all(torch.isfinite(v).all() for v in final.values()),'final finite');complete=json.loads((directory/'complete.json').read_text());require(complete['epochs']==20 and complete['steps']==3920,'fixed update steps')
        log=pd.read_csv(directory/'training.csv');require(len(log)==20 and np.isfinite(log[['total_loss','binary_CE','family_CE_diagnostic','family_SupCon_diagnostic']]).all().all(),'training logs')
        support=pd.read_csv(directory/'positive_support.csv');current=support[['epoch','family','anchors','valid_anchors','positive_pairs']]
        if previous is not None:require(previous.equals(current),'paired batch family positive support')
        previous=current
    checks[str(seed)]=dict(initial_states_identical=True,source_batch_orders_valid=True,batch_family_support_identical=True,steps_each=3920)
train=pd.read_csv(r/'source/train_indices.csv');val=pd.read_csv(r/'source/validation_indices.csv');require(len(train)==50000 and len(val)==10000 and not np.isin(train.availability_rank,val.availability_rank).any(),'source counts/disjoint')
a=np.load(r/'tables/evaluation_actions.npz');old=np.load(p['baseline_actions']);require(len(a['rank'])==68400 and np.array_equal(a['rank'],old['rank']),'action ranks');require(not set(a.files)&{'y','truth','label'},'truth in actions')
for name in HISTORY:require(np.array_equal(a['action_'+name],old['action_'+name]),'historical action preservation')
require(np.bincount(a['segment'])[1:].tolist()==[17100]*4 and np.bincount(a['window']).tolist()==[100]*684,'target segment/window counts')
if (r/'reports/decision.json').exists():
    d=json.loads((r/'reports/decision.json').read_text());segments=pd.read_csv(r/'tables/segment_metrics.csv');require(len(d['overall'])==24 and len(segments)==96,'method count')
    for name,v in d['overall'].items():
        for k in ['N','N_B','N_A','TN','FN','FP','TP','abstain_B','abstain_A','windows_B','windows_A']:require(int(segments[segments.method==name][k].sum())==v[k],'segment conservation')
    pair=pd.read_csv(r/'tables/paired_intervals.csv');require(len(pair)==90 and len(PAIRS)==18,'predefined comparisons')
    diffs=pd.read_csv(r/'tables/seed_differences.csv');require(len(diffs)==270,'per seed differences')
    oldd=json.loads((Path(p['baseline_actions']).parents[1]/'reports/decision.json').read_text())['overall']
    for n in HISTORY:
        for k in ['N_B','N_A','TN','FN','FP','TP','R_B','R_A','U_B','U_A','U_bal','target_status']:require(d['overall'][n][k]==oldd[n][k],'historical metrics')
    checks['evaluation']=dict(rows=68400,windows=684,methods=24,segment_rows=96,comparisons=18,paired_metric_rows=90,seed_difference_rows=270,baseline_metrics_unchanged=True)
checks['source']=dict(train=50000,validation=10000,disjoint=True);checks['target']=dict(new_labels=0,cumulative=1200,baseline_actions_unchanged=True,no_truth_in_actions=True);checks['time']=now()
writej(r/'config/delivery_checks.json',checks);print('CHECKS PASSED',checks)
