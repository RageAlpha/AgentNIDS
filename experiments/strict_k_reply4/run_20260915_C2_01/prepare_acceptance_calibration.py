from common import *
import copy
import pyarrow.parquet as pq
sys.path.insert(0,str(ROOT/'src'))
from nids.data.partitions import PreparedData
from nids.evaluation.outer_data import checked_windows

assert not (OUT/'protocol.json').exists()
write(OUT/'upstream_inventory_before.json',inventory())
assert sha(PLAN)==PLAN_SHA
PLAN.read_text()
for p in [PRE/'C_preflight_report.md',C1/'stage_C1_report.md',B2/'scope_and_fit_definition.md',
    ROOT/'outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01/scope_and_score_definition.md']:
    p.read_text()
for fn in ['protocol_freeze.json','threshold_freeze.json']:
    for p,h in read(C1/fn)['files'].items(): assert sha(p)==h
for p,h in read(C1/'protocol.json')['preflight_file_sha256'].items(): assert sha(p)==h
for n,h in read(PRE/'rank_commitment.json')['sha256'].items(): assert sha(PRE/n)==h
assert read(C1/'validation.json')['status']=='PASS_C1_COMPLETE_STOPPED_FOR_REVIEW'
prep=PreparedData(ROOT/'outputs/round1_data_prep_v1')
dev=rows(PRE/'development_R_select_scope.csv'); refs=[]; audits=[]
sample=rows(PRE/'calibration_sample_manifest.csv')
for d,(fold,end,nw,nb,na) in DOMAINS.items():
    part,ws=checked_windows(prep,d,pq.read_table(ROOT/f'outputs/shift_v2/10k/{d}_windows.parquet').to_pylist())
    assert ws[9]['rank_end_exclusive']==100000 and ws[-1]['rank_end_exclusive']==end and len(ws)-10==nw
    sr=[r for r in sample if r['domain']==d]; rank=np.array([int(r['availability_rank']) for r in sr])
    assert np.array_equal(rank,np.load(PRE/f'{d}_sorted_ranks.npy')) and len(set(rank))==400
    order=prep.sorted_rows(d); orig=np.array([int(r['original_row']) for r in sr])
    assert np.array_equal(order[rank],orig) and max(rank)<100000
    assert (sum(int(r['binary_label'])==0 for r in sr),sum(int(r['binary_label'])==1 for r in sr))==(nb,na)
    overlap=sum(np.isin(order[a:min(a+1000000,end)],orig).sum() for a in range(100000,end,1000000))
    assert overlap==0
    audits.append(dict(domain=d,identity_intersection=int(overlap),sample_rows=400))
    bm=read(C1/f'bootstrap/{d}_manifest.json')
    ix=C1/f'bootstrap/{d}_sampled_blocks.npy'; mp=C1/f'bootstrap/{d}_block_multiplicity.npy'
    assert sha(ix)==bm['sampled_blocks_sha256'] and sha(mp)==bm['block_multiplicity_sha256']
    idx=np.load(ix); mul=np.load(mp); blocks=(nw+9)//10
    assert idx.shape==mul.shape==(2000,blocks)
    assert all(np.array_equal(np.bincount(i,minlength=blocks),m) for i,m in zip(idx,mul))
    br=rows(C1/f'bootstrap/{d}_block_scope.csv')
    assert len(br)==blocks and int(br[0]['rank_start'])==100000 and int(br[-1]['rank_end_exclusive'])==end
    for i,r in enumerate(br):
        assert int(r['rank_start'])==100000+i*100000 and int(r['rank_end_exclusive'])==min(end,200000+i*100000)
    oldwindows=[r for r in rows(C1/'evaluation_window_references.csv') if r['domain']==d]
    assert [r['window_id'] for r in oldwindows]==[w['window_id'] for w in ws[10:]]
    for p in [ix,mp,C1/f'bootstrap/{d}_manifest.json',C1/f'bootstrap/{d}_block_scope.csv']:
        refs.append(dict(domain=d,path=str(p),sha256=sha(p),source='C1 shifted evaluation scope',copied=False))
for r in dev:
    ws=sorted([w for w in prep.windows if w['domain']==r['domain'] and w['scheme']=='development' and w['partition']=='R_select' and int(w['window_size'])==10000],key=lambda w:int(w['rank_start']))
    assert int(ws[0]['rank_start'])==int(r['rank_start']) and int(ws[-1]['rank_end_exclusive'])==int(r['rank_end_exclusive'])
    assert len(ws)==int(r['complete_windows'])
    for i,w in enumerate(ws): assert int(w['rank_start'])==int(r['rank_start'])+i*10000
save(OUT/'bootstrap_sources.csv',refs); write(OUT/'identity_validation.json',audits)
assets=[]; cfg=read(ROOT/'configs/execution_v2.json'); oldassets=rows(PRE/'asset_availability.csv')
for d,(fold,*_) in DOMAINS.items():
    for m in MODELS:
        cache(d,m)
        for kind in ['detector','Platt']:
            r=next(r for r in oldassets if r['kind']==kind and r['model']==m)
            assert sha(r['path'])==r['sha256']
            assets.append(dict(kind=kind,path=r['path'],sha256=r['sha256'],model=m))
        for o in FIELDS:
            p=B2/f'models/{fold}__{m}__{o}.json'; fit=read(p)
            assert (fit['fold'],fit['model'],fit['output'],fit['target'])==(fold,m,o,d)
            assert fit['development']==cfg['folds'][fold]['development'] and d not in fit['development']
            assert fit['status']=='COMPLETE' and fit['fit_partition']=='R_fit' and fit['scale']>0
            old=next(r for r in oldassets if r['kind']=='frozen_fu' and r['path']==str(p))
            assert sha(p)==old['sha256']
            fr=next(r for r in rows(B2/'fit_manifest.csv') if (r['fold'],r['model'],r['output'])==(fold,m,o))
            assert float(fr['beta0'])==fit['beta0'] and float(fr['beta1'])==fit['beta1']
            assets.append(dict(kind='fu',path=str(p),sha256=sha(p),domain=d,fold=fold,model=m,output=o,development=fit['development']))
definition=dict(assets=assets,B2_code=str(B2/'run_b2.py'),B2_code_sha256=sha(B2/'run_b2.py'),
    classification='p>=0.5 ALWAYS; no C1 classification threshold read for decisions',
    entropy='direct frozen B2.entropy: cupyx.scipy.special.xlogy, float64 CUDA; 0log0=0',
    fu='direct frozen B2.predict: torch float64 CUDA sigmoid(beta0+beta1*(u-anchor-mean_offset)/scale)',
    score_direction='s>=a; high scores first; no clipping, rounding, reversal or tie splitting',
    full_accept_boundary=ACCEPT_ALL,full_reject_sentinel=REJECT_ALL,score_range=[0,1])
write(OUT/'score_definition_and_assets.json',definition)
protocol=copy.deepcopy(read(PRE/'protocol_draft.json'))
protocol['historical_preflight_prohibitions']=protocol.pop('prohibited_this_run')
protocol.update(version='strict-k-reply4-C2-v1.0',status='FROZEN_FOR_C2_AND_C0_SELECT',frozen_at_utc=now(),
    source_plan_sha256=PLAN_SHA,source_preflight=str(PRE),source_C1=str(C1),
    user_request_path='/home/karri/.codex/attachments/25492af7-10fb-43d9-936a-ebbb3289630e/pasted-text.txt',
    preflight_file_sha256={str(p):sha(p) for p in PRE.iterdir() if p.is_file()},
    C1_input_sha256={str(p):sha(p) for p in C1.rglob('*') if p.is_file()},
    executed_scope=['36 C0-select','24 C2-select; 12 unsupported CIC placeholders','fixed p>=0.5 all-accept background','evaluation and paired bootstrap'],
    current_prohibitions=['C1 classification thresholds','joint search','calibration resampling or new labels','detector/Platt/fu fitting','K sensitivity','conformal','coordination','open-set'],
    classification_threshold=0.5,numerical_definition=definition,
    C0_exact_selection=dict(N1N2_bound='40*N1*N2 < int64 max checked before arithmetic',
        weighted_accept_numerator='CA=A1*N2+A2*N1',weighted_error_numerator='CE=E1*N2+E2*N1',
        coverage='CA/(2*N1*N2)',risk='CE/CA',feasible='5*CA>=N1*N2 and 20*CE<=CA',
        optimization='maximize CA; then minimize CE/CA; then higher score threshold',
        candidates='all distinct exact score groups in pooled score union + -1 and 2',
        audit='all-group chunk digests, feasible runs, endpoint counts and selected sufficient statistics; no approximate candidate pruning'),
    bootstrap_execution=read(C1/'protocol.json')['bootstrap_execution'],bootstrap_action='reuse C1 matrices by verified absolute reference, no draws',
    temporal_segments=SEGMENTS,full_accept_boundary=-1.0,full_reject_sentinel=2.0,
    target_label_boundary='all C0/C2 thresholds frozen before target evaluation; development domains have historical cross-fold target roles; no claim of never-seen labels')
protocol['C2']['status']='AUTHORIZED_NOT_YET_EXECUTED'; protocol['C0_select']['status']='AUTHORIZED_NOT_YET_EXECUTED'
protocol['C1']['status']='HISTORICAL_COMPLETE_NOT_EXECUTED_THIS_RUN'
write(OUT/'protocol.json',protocol)
write(OUT/'protocol_freeze.json',dict(frozen_at_utc=now(),files={str(OUT/n):sha(OUT/n) for n in ['protocol.json','score_definition_and_assets.json']},
    request_sha256=sha(protocol['user_request_path']),target_evaluation_labels_read=False))
log('Protocol and scores/assets frozen. Original sample, C1 hashes and all bootstrap block identities verified.')
