"""C-preflight only. No probability values, threshold search, fitting, or evaluation labels."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path('/home/karri/projects/NIDSProject_Professor_Run')
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'src'))
from nids.data.partitions import PreparedData
from nids.evaluation.outer_data import checked_windows

DOMAINS = {'ToN': ('F1', 27520000, 260), 'CIC': ('F3', 20110000, 5529), 'BoT': ('F2', 16930000, 3808)}
B2 = ROOT/'outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01'
SEED, K, STARTUP = 20260914, 400, 100000
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def save(name, data):
    keys = list(dict.fromkeys(k for r in data for k in r))
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows({k: 'NA' if r.get(k) is None else r[k] for k in keys} for r in data)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(s): print(now(), s, flush=True)
def inventory():
    result = {}
    for folder in ['configs', 'docs', 'src', 'datasets', 'outputs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts:
                s = p.stat()
                result[str(p)] = {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}
                if s.st_size <= 2000000 and p.suffix in ['.json', '.md', '.py', '.csv', '.sh']:
                    result[str(p)]['sha256'] = sha(p)
    return result

def main():
    started = now()
    assert not (OUT/'calibration_ranks.csv').exists(), 'Never overwrite committed rank draw'
    before = inventory(); write('upstream_inventory_before.json', before)
    cfg = read(ROOT/'configs/execution_v2.json')
    prep = PreparedData(Path(cfg['prepared_run']))
    plan = ROOT/'docs/StrictK_Reply4_Experiment_Plan.md'
    plan_text = plan.read_text()
    assert 'PCG64' in plan_text and '64260000' in plan_text
    evidence = [plan, ROOT/'configs/execution_v2.json', ROOT/'configs/p7_v1.json',
        ROOT/'outputs/round1_v2/config_snapshot/freeze_record_revision_v2_1.json',
        prep.root/'split_manifest.csv', prep.root/'window_manifest.csv', prep.root/'tail_manifest.csv',
        prep.root/'fold_manifest.json', prep.root/'round1_config.json', prep.root/'artifact_validation.json',
        ROOT/'outputs/source_v2/P2/P2_manifest.json', ROOT/'outputs/round1_v2/P3/P3_manifest.json',
        ROOT/'outputs/outer_v2/P7/P7_manifest.json', B2/'fits_frozen.json', B2/'fit_config.json', B2/'fit_manifest.csv']
    for stage, run, names in [
        ('stage_A','run_20260914_A1',['stage_A_report.md','asset_and_scope.md','evaluation_scope.csv','model_asset_audit.csv','final_validation.json']),
        ('stage_B1','run_20260914_B1_01',['stage_B1_report.md','scope_and_score_definition.md','evaluation_scope.csv','validation.json']),
        ('stage_B2','run_20260914_B2_01',['stage_B2_report.md','scope_and_fit_definition.md','evaluation_scope.csv','validation.json'])]:
        evidence += [ROOT/'outputs/post_p7_diagnostics'/stage/run/n for n in names]
    for p in evidence: p.read_text()
    save('source_evidence.csv', [{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in evidence])
    windows, scope = {}, []
    folds = read(prep.root/'fold_manifest.json')
    for d, (fold, end, tail) in DOMAINS.items():
        assert cfg['folds'][fold]['heldout'] == folds[fold]['heldout'] == d
        assert cfg['folds'][fold]['development'] == folds[fold]['development']
        part, ws = checked_windows(prep, d, pq.read_table(ROOT/f'outputs/shift_v2/10k/{d}_windows.parquet').to_pylist())
        assert part.rank_start == 0 and ws[-1]['rank_end_exclusive'] == end
        assert ws[9]['rank_end_exclusive'] == STARTUP and ws[10]['rank_start'] == STARTUP
        assert part.rank_end_exclusive-end == tail
        old = read(ROOT/f'outputs/outer_v2/P7/{fold}/data_audit.json')
        assert (old['rows'], old['windows'], old['excluded_tail']) == (end, len(ws), tail)
        for p in evidence:
            if p.name == 'evaluation_scope.csv':
                r = next(r for r in rows(p) if r['domain']==d and r['partition']=='T_heldout')
                assert int(r['rows']) == end and int(r['excluded_tail']) == tail
        windows[d] = ws
        ends = prep.sorted_end_ms(d)
        scope.append(dict(domain=d, fold=fold, startup_rank_start=0, startup_rank_end_exclusive=STARTUP,
            startup_rows=STARTUP, startup_complete_windows=10, unlabelled_startup_excluded=99600,
            evaluation_rank_start=STARTUP, evaluation_rank_end_exclusive=end, evaluation_rows=end-STARTUP,
            evaluation_complete_windows=len(ws)-10, old_excluded_tail=tail, tail_rank_start=end,
            tail_rank_end_exclusive=part.rank_end_exclusive, first_startup_window_id=ws[0]['window_id'],
            last_startup_window_id=ws[9]['window_id'], first_evaluation_window_id=ws[10]['window_id'],
            last_evaluation_window_id=ws[-1]['window_id'], first_available_ms=int(ends[0]),
            startup_last_available_ms=int(ends[STARTUP-1]), evaluation_first_available_ms=int(ends[STARTUP]),
            startup_wait_ms=int(ends[STARTUP-1])-int(ends[0]),
            startup_wait_seconds=(int(ends[STARTUP-1])-int(ends[0]))/1000,
            boundary_same_availability_batch=bool(ends[STARTUP-1]==ends[STARTUP]),
            wait_definition='FLOW_END_MILLISECONDS[rank 99999] - FLOW_END_MILLISECONDS[rank 0]; availability proxy, not wall-clock annotation latency'))
    assert sum(r['evaluation_rows'] for r in scope) == 64260000
    assert sum(r['evaluation_complete_windows'] for r in scope) == 6426
    save('evaluation_scope.csv', scope)
    # Rank-only commitment for ALL domains occurs before opening any target label array.
    draws, rankrows = {}, []
    for d in DOMAINS:
        rng = np.random.Generator(np.random.PCG64(SEED))
        draw = rng.choice(STARTUP, size=K, replace=False)
        draws[d] = np.sort(draw)
        np.save(OUT/f'{d}_draw_order_ranks.npy', draw)
        np.save(OUT/f'{d}_sorted_ranks.npy', draws[d])
        rankrows += [dict(domain=d, availability_rank=int(r)) for r in draws[d]]
    save('calibration_ranks.csv', rankrows)
    commitment = dict(committed_at_utc=now(), labels_opened=False, probabilities_read=False,
        algorithm='per-domain numpy.random.Generator(numpy.random.PCG64(20260914)).choice(100000, size=400, replace=False); sort after draw',
        numpy_version=np.__version__, sha256={p.name:sha(p) for p in OUT.glob('*ranks*') if p.is_file()})
    write('rank_commitment.json', commitment)
    log('All three rank draws saved and hashed; no target labels opened.')
    identities, validations = {}, []
    # Full identity ledger audit reads only identity/time columns, never target labels.
    for d, (fold, end, tail) in DOMAINS.items():
        order, ends, selected = prep.sorted_rows(d), prep.sorted_end_ms(d), draws[d]
        seen = np.zeros(len(order), dtype=bool)
        for a in range(0, len(order), 262144):
            v = np.asarray(order[a:a+262144])
            assert (v>=0).all() and (v<len(order)).all()
            assert len(np.unique(v)) == len(v) and not seen[v].any()
            seen[v] = True
            if a: assert ends[a]>=ends[a-1]
            t = ends[a:a+len(v)]
            assert (t[1:]>=t[:-1]).all()
            assert (v[1:][t[1:]==t[:-1]] > v[:-1][t[1:]==t[:-1]]).all()
            if a and ends[a]==ends[a-1]: assert order[a]>order[a-1]
        assert seen.all(); del seen
        cursor, picked = 0, []
        path = ROOT/f'outputs/outer_v2/P7/{fold}/identities.parquet'
        pf = pq.ParquetFile(path)
        assert pf.metadata.num_rows == end
        for batch in pf.iter_batches(batch_size=262144, columns=['rank','original_row','record_id','availability_ms','window_id','complete_window']):
            stop = cursor+batch.num_rows
            rr = batch.column('rank').to_numpy()
            assert np.array_equal(rr, np.arange(cursor,stop))
            assert np.array_equal(batch.column('original_row').to_numpy(), order[cursor:stop])
            assert np.array_equal(batch.column('availability_ms').to_numpy(), ends[cursor:stop])
            expected = pc.binary_join_element_wise(d+':', pc.cast(batch.column('original_row'), pa.string()), '')
            assert pc.all(pc.equal(expected,batch.column('record_id'))).as_py()
            assert pc.all(batch.column('complete_window')).as_py()
            wi = rr//10000
            for index in np.unique(wi):
                lo=max(cursor,int(index)*10000)-cursor; hi=min(stop,(int(index)+1)*10000)-cursor
                assert pc.all(pc.equal(batch.column('window_id').slice(lo,hi-lo), windows[d][int(index)]['window_id'])).as_py()
            take = selected[(selected>=cursor)&(selected<stop)]-cursor
            if len(take): picked += pa.Table.from_batches([batch]).take(pa.array(take)).to_pylist()
            cursor=stop
        assert cursor==end and len(picked)==K
        assert len({r['record_id'] for r in picked})==K
        identities[d]=picked
        validations.append(dict(domain=d,full_identity_rows_checked=end,sorted_order_is_permutation=True,
            stable_availability_order=True,rank_original_row_record_id_time_window_match=True,
            sampled_unique=K,all_sample_ranks_in_startup=True,calibration_evaluation_identity_intersection=0,
            disjoint_proof='full rank-to-original-row permutation + complete P7 identity ledger + disjoint half-open rank intervals'))
        log(f'{d}: full {end:,}-row identity ledger verified; calibration/evaluation disjoint.')
    # Read only selected source CSV records. Other source lines are skipped as bytes,
    # never decoded or split into labels; no startup label histogram is constructed.
    manifest, support, windowcounts, categorycounts, label_audit = [], [], [], [], []
    for d in DOMAINS:
        source = Path(cfg['data_root'])/cfg['datasets'][d]
        wanted = {r['original_row']:r for r in identities[d]}
        originals = {}
        with source.open('rb') as f:
            header=next(csv.reader([f.readline().decode('utf-8-sig')]))
            for index, line in enumerate(f):
                if index in wanted:
                    fields=next(csv.reader([line.decode('utf-8')]))
                    originals[index] = dict(zip(header,fields))
                if len(originals)==K: break
        assert len(originals)==K
        label_open = now()
        lab = np.load(ROOT/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')
        code = np.load(ROOT/f'outputs/feature_store_v2/{d}_attack_code.npy',mmap_mode='r')
        book = read(ROOT/f'outputs/feature_store_v2/{d}_manifest.json')['attack_codebook']
        y = np.array(lab[draws[d]]); codes = np.array(code[draws[d]])
        assert np.isin(y,[0,1]).all()
        for i,r in enumerate(identities[d]):
            raw=originals[r['original_row']]
            assert int(raw['Label'])==int(y[i]) and raw['Attack']==book[str(int(codes[i]))]
            assert int(raw['FLOW_END_MILLISECONDS'])==r['availability_ms']
            manifest.append(dict(domain=d, availability_rank=r['rank'], original_row=r['original_row'],
                record_id=r['record_id'], window_id=r['window_id'], availability_ms=r['availability_ms'],
                FLOW_START_MILLISECONDS=raw['FLOW_START_MILLISECONDS'], FLOW_END_MILLISECONDS=raw['FLOW_END_MILLISECONDS'],
                binary_label=int(y[i]), original_attack_category=raw['Attack'], original_attack_code=int(codes[i]),
                identity_definition='existing P7 domain:zero-based original CSV data row; header excluded',
                native_record_id=None, native_record_id_note='source CSV has no record_id; retained existing P7 identity',
                missing_time_fields='none: both original FLOW_START/END_MILLISECONDS present'))
        counts={}
        for r in identities[d]: counts[r['window_id']]=counts.get(r['window_id'],0)+1
        for w in windows[d][:10]:
            windowcounts.append(dict(domain=d,window_id=w['window_id'],rank_start=w['rank_start'],rank_end_exclusive=w['rank_end_exclusive'],sample_count=counts.get(w['window_id'],0)))
        cats={}
        for c in codes: cats[book[str(int(c))]]=cats.get(book[str(int(c))],0)+1
        for category,count in cats.items(): categorycounts.append(dict(domain=d,original_attack_category=category,count=count))
        benign, attack = int((y==0).sum()), int((y==1).sum())
        ok=benign>=20 and attack>=20
        support.append(dict(domain=d,sample_size=K,unique_ranks=K,unique_record_ids=K,all_in_startup=True,
            benign=benign,attack=attack,windows_covered=len(counts),minimum_per_class=20,label_support_sufficient=ok,
            support_status='SUFFICIENT' if ok else 'TARGET_CLASS_SUPPORT_INSUFFICIENT',
            window_sample_counts_json=json.dumps(counts),original_attack_category_counts_json=json.dumps(cats),
            sampling_changed=False))
        label_audit.append(dict(domain=d,rank_commitment_precedes_source_label_read=True,
            first_feature_label_open_utc=label_open,source_labels_decoded=K,feature_labels_indexed=K,
            distinct_target_labels_read=K,unselected_startup_labels_read=0,evaluation_labels_read=0,
            source_and_feature_labels_and_categories_match=True))
        log(f'{d}: fixed sample Benign={benign}, Attack={attack}, status={support[-1]["support_status"]}')
    save('calibration_sample_manifest.csv',manifest); save('calibration_support.csv',support)
    save('calibration_window_counts.csv',windowcounts); save('calibration_attack_category_counts.csv',categorycounts)
    write('label_access_audit.json',label_audit)
    assets=[]
    def asset(kind,path,**meta):
        p=Path(path); assert p.is_file() and p.stat().st_size>0
        assets.append(dict(kind=kind,path=str(p),status='AVAILABLE_METADATA_VERIFIED',bytes=p.stat().st_size,**meta))
    models=cfg['models']['schedule']; assert len(models)==6
    for m in models:
        p=ROOT/f'outputs/source_v2/models/{m}/training_manifest.json'; train=read(p)
        model=p.parent/train['artifact']; assert train['model_id']==m and train['status']=='COMPLETE'
        assert train['training_partition']=='S_train' and model.stat().st_size==train['artifact_size_bytes']
        asset('detector',model,model=m,feature_count=train['feature_count'],metadata_path=str(p),fit_domain='UNSW',fit_partition='S_train',sha256=sha(model))
        calpath=ROOT/f'outputs/source_v2/calibrators/{m}.json'; cal=read(calpath)
        assert cal['model_id']==m and cal['fit_domain']=='UNSW' and cal['fit_partition']=='S_prob_cal'
        assert cal['status']=='COMPLETE' and np.isfinite([cal['coefficient'],cal['intercept']]).all()
        asset('Platt',calpath,model=m,fit_domain='UNSW',fit_partition='S_prob_cal',sha256=sha(calpath))
        for d in DOMAINS:
            path=ROOT/f'outputs/prediction_cache_v2/{d}/{m}.npy'
            cm=read(path.with_name(m+'_manifest.json')); dm=read(path.parent/'manifest.json')
            cache=np.load(path,mmap_mode='r')
            assert cm['model_id']==m and cm['domain']==d and cm['feature_indices']==train['feature_indices']
            assert cm['feature_names']==train['feature_names'] and cm in dm['models']
            assert dm['row_order']=='availability_rank' and Path(dm['original_row_index'])==prep.root/f'{d}_sorted_original_rows.npy'
            assert len(cache)==cm['rows']==dm['rows']==len(prep.sorted_rows(d))
            assert path.stat().st_size==cm['bytes']==cache.offset+cache.nbytes
            assert cache.dtype.names==('raw_probability','probability','predicted_class','u','h')
            for output,field in [('Raw','raw_probability'),('Platt','probability')]:
                assert cache.dtype[field]==np.dtype('float64')
                asset('probability_cache',path,domain=d,fold=DOMAINS[d][0],model=m,output=output,
                    field=field,rows=len(cache),metadata_path=str(path.with_name(m+'_manifest.json')),
                    verification='read-only mmap header, byte size, full rank identity contract, model/feature metadata; no probability values read',
                    usable_for='C1; C2/C0-select if All49')
    devscope=[]
    oldscope=rows(B2/'evaluation_scope.csv'); fitrows=rows(B2/'fit_manifest.csv')
    for fold,roles in cfg['folds'].items():
        d=roles['heldout']
        for dev in roles['development']:
            part=prep.partition(dev,'development','R_select')
            ws=sorted([w for w in prep.windows if w['domain']==dev and w['scheme']=='development' and w['partition']=='R_select' and int(w['window_size'])==10000],key=lambda w:int(w['rank_start']))
            cursor=part.rank_start
            for w in ws:
                a,b=int(w['rank_start']),int(w['rank_end_exclusive'])
                assert a==cursor and b-a==10000
                assert (int(w['first_original_row']),int(w['last_original_row']))==(int(prep.sorted_rows(dev)[a]),int(prep.sorted_rows(dev)[b-1]))
                cursor=b
            old=next(r for r in oldscope if r['domain']==dev and r['partition']=='R_select')
            assert cursor==int(old['rank_end_exclusive']) and part.rank_start==int(old['rank_start'])
            assert 0<=part.rank_end_exclusive-cursor<10000
            devscope.append(dict(fold=fold,target=d,domain=dev,partition='R_select',rank_start=part.rank_start,
                rank_end_exclusive=cursor,rows=cursor-part.rank_start,complete_windows=len(ws),
                excluded_tail=part.rank_end_exclusive-cursor,domain_weight=0.5,label_cache_metadata_available=True,
                probability_cache_available=True,labels_read=False))
            asset('R_select_label_cache',ROOT/f'outputs/feature_store_v2/{dev}_label.npy',domain=dev,fold=fold,
                rows=len(np.load(ROOT/f'outputs/feature_store_v2/{dev}_label.npy',mmap_mode='r')),
                rank_start=part.rank_start,rank_end_exclusive=cursor,verification='header only; no development labels read')
        for m in [m for m in models if m.endswith('All49')]:
            for output in ['Raw','Platt']:
                p=B2/f'models/{fold}__{m}__{output}.json'; fit=read(p)
                assert (fit['fold'],fit['model'],fit['output'],fit['target'])==(fold,m,output,d)
                assert fit['development']==roles['development'] and d not in fit['development']
                assert fit['fit_partition']=='R_fit' and fit['status']=='COMPLETE' and fit['converged_gradient']
                assert fit['scale']>0 and np.isfinite([fit[k] for k in ['anchor','mean_offset','scale','beta0','beta1']]).all()
                assert [a['domain'] for a in fit['training_audits']]==roles['development']
                for a in fit['training_audits']:
                    part=prep.partition(a['domain'],'development','R_fit')
                    assert a['partition']=='R_fit' and (a['rank_start'],a['rank_end_exclusive'])==(part.rank_start,part.rank_end_exclusive)
                fr=next(r for r in fitrows if (r['fold'],r['model'],r['output'])==(fold,m,output))
                assert float(fr['beta0'])==fit['beta0'] and float(fr['beta1'])==fit['beta1']
                asset('frozen_fu',p,domain=d,fold=fold,model=m,output=output,sha256=sha(p),
                    development=json.dumps(roles['development']),fit_partition='R_fit',
                    field='raw_probability' if output=='Raw' else 'probability',
                    verification='complete finite frozen coefficients, scale>0, convergence, fold/model/output and fit_manifest match',
                    formula='sigmoid(beta0 + beta1 * (u-anchor-mean_offset)/scale); original B2 float64 CUDA arithmetic')
    assert sum(a['kind']=='frozen_fu' for a in assets)==18
    save('asset_availability.csv',assets); save('development_R_select_scope.csv',devscope)
    protocol=dict(status='DRAFT_AFTER_C_PREFLIGHT_AWAITING_USER_REVIEW',created_at_utc=now(),
        design_context='Exploratory design proposed after inspecting A/B1/B2 and already unblinded target domains; not retrospective old preregistration or new unseen-domain validation',
        source_plan=str(plan),source_plan_sha256=sha(plan),K=K,per_domain_label_budget=K,
        startup=dict(partition='P7 T_heldout',first_complete_windows=10,window_size=10000,rank_interval=[0,STARTUP],interval_convention='half-open',exclude_entire_startup_from_evaluation=True),
        sampling=dict(generator='numpy.random.Generator',bit_generator='PCG64',seed=SEED,new_generator_per_domain=True,
            call='choice(100000, size=400, replace=False)',sort_after_sampling=True,commit_before_label_read=True,
            shared_sample_across_all_methods=True,resampling=False,replenishment=False,class_balancing=False,numpy_version=np.__version__),
        support=dict(benign_min=20,attack_min=20,failure='TARGET_CLASS_SUPPORT_INSUFFICIENT',
            consequence='retain domain report and C0; no deployable C1 or C2 threshold; label support limitation, not model performance failure'),
        evaluation=scope,folds=cfg['folds'],models_C1=models,models_C2=[m for m in models if m.endswith('All49')],outputs=['Raw','Platt'],
        C0_class=dict(classification='p>=0.5',acceptance='all',recompute_only_after_approval=True),
        C1=dict(status='NOT_RUN',configurations=36,classification='p>=t_K',acceptance='all',
            candidates='unique 400 calibration probabilities plus 0, 0.5, nextafter(1,+infinity)',
            constraint='20*FP<=N_Benign (empirical FPR<=5%); integer ratio comparisons',
            order=['maximize Attack recall','minimize FPR','minimize abs(t-0.5)','maximize t'],
            ties='whole equal-probability group',include_calibration_t_half_control=True,
            zero_optimal_recall='NO_NONTRIVIAL_CLASSIFICATION_SOLUTION; retain degenerate point, no automatic fallback to 0.5',
            save_threshold_before_evaluation_labels=True),
        C2=dict(status='NOT_RUN',configurations=36,classification='p>=0.5',scores=['1-u','own-fold frozen B2 f(u)'],
            entropy='-(xlogy(p,p)+xlogy(1-p,1-p))/ln(2); float64; 0log0=0; no extra clipping/rounding/reversal',
            fu='sigmoid(beta0+beta1*(u-anchor-mean_offset)/scale); reuse B2 implementation',
            candidates='unique 400 scores plus full-accept boundary and full-reject sentinel',acceptance='s>=a; whole ties',
            constraint='20*accepted_errors<=accepted_rows and accepted_rows>=40',
            order=['maximize coverage','minimize empirical selective risk','maximize threshold'],
            no_solution='NO_FEASIBLE_ACCEPTANCE_THRESHOLD; full reject: coverage=0, risk=NA, effective_attack_detection=0; not a safety success',
            support_gate='overall 400 records Benign>=20 and Attack>=20; no accepted-subclass gate',
            finite_sample_risk_guarantee=False),
        C0_select=dict(status='NOT_RUN',scope=devscope,configurations=36,classification='p>=0.5',
            candidates='union of unique scores from both complete R_select scopes plus full accept/reject sentinels',
            coverage='(c_1+c_2)/2',risk='(e_1+e_2)/(c_1+c_2); e_d=accepted errors / all domain records',
            constraint='empirical risk<=5%, coverage>=10%',order=['maximize coverage','minimize risk','maximize threshold'],
            no_solution='full reject; risk NA',baseline_is_new=True,equal_label_budget_causal_comparison=False),
        information_boundary=dict(detectors_and_Platt='frozen UNSW source training/calibration',fu='frozen own-fold development R_fit only',
            target_labels_used_for_retraining=False,probability_values_read_in_preflight=False,
            cross_fold_role_reuse='historical target records may be R_fit in other folds; preserve roles; not new independent samples'),
        later_evaluation_rules=dict(temporal_segments='four contiguous nearly equal window segments, early segments receive remainder; fixed threshold',
            bootstrap='new shared per-domain 2000 paired resamples of 10 consecutive complete windows; retain short last block; >=20 blocks; never reuse matrices containing startup',
            undefined_rates='NA; report valid replicate counts',interval_scope='conditional on one 400-label draw, frozen models and current domains',
            metrics_reference='source plan sections 8-9; C1 FPR/Recall/BA and C2 coverage/risk/attack outcomes and native categories; do not select winners by alternate endpoints'),
        approval_sequence=['C-preflight review','C1 separately authorized','C1 review','C2 and C0-select separately authorized'],
        prohibited_this_run=['threshold search/selection','C1','C2','C0 performance','FPR/Recall/risk/AURC on evaluation records','bootstrap generation','Strict-K sensitivity','conformal','coordination','open-set','any refitting'])
    write('protocol_draft.json',protocol)
    after=inventory(); write('upstream_inventory_after.json',after)
    assert before==after, 'Upstream files changed'
    assert all(sha(OUT/name)==digest for name,digest in commitment['sha256'].items())
    validation=dict(status='PASS_C_PREFLIGHT_COMPLETE',started_at_utc=started,completed_at_utc=now(),
        identity_checks=validations,label_access=label_audit,total_calibration_records=len(manifest),
        total_evaluation_records=64260000,total_evaluation_complete_windows=6426,probability_cache_conditions=36,
        frozen_fu_models=18,all_assets_metadata_available=True,upstream_files_unchanged=len(before),
        preservation_check='all upstream files size/mtime; SHA256 additionally for <=2MB JSON/MD/PY/CSV/SH; six detector artifacts and 18 f(u) hashes in asset CSV',
        rank_commitment_unchanged=True,no_thresholds_fitted_or_selected=True,no_evaluation_labels_read=True,
        no_probability_values_read=True,no_performance_metrics_computed=True,protocol_data_conflicts=[],
        limitations=['metadata availability does not constitute new full probability value validation; reuse A/B1/B2 audit evidence',
            'native record_id absent; existing P7 identity retained',
            'already unblinded targets and historical cross-fold reuse do not become independent unseen validation'],
        domain_support=support,C1_started=False,C2_started=False,user_review_required_before_C1=True,
        versions=dict(python=sys.version,numpy=np.__version__,pyarrow=pa.__version__))
    write('validation.json',validation)
    lines=['# C-preflight: Fixed 400 target calibration sample and asset verification','',
        'Pre-checking is completed, identity / window / quantity and asset metadata verification is passed. This time, only the pre-test material was generated without reading the probability value, searching or selecting the threshold value, calculating subsequent performance, running C1/C2. The labels support is insufficient to report separately and is not considered a model performance failure.','',
        f' Project root directory: `{ROOT}`. Program directly reading `{plan}`，SHA256 `{sha(plan)}`. Output directory: `{OUT}`。',
        'The scheme is based on the exploratory design of the A/B1/B2 and dates back to pre-registration. target domain has been unblinding, excluding the calibration record to avoid direct overlapping of the wheel, and cannot be restored to a completely new unseen field validation.','',
        '## sampling with labels support','',
        'complete windows and the original identity rules are verified with checked_windows, then Generator(PCG64(20260914)) is reconstructed for each domain, choice(100000,size=400,replace=False) is called, and then reads sequentially. All three rank domains are saved and hashed to read any labels in the drawing. The same seeds and candidate numbers in each domain produce the same rank sequence, as a result of the specified algorithm; The domain and the original line identity are different. Each domain only decodes the original CSV labels of the 400 in the abstract and verifies with the original labels and category in the cache abstract; There are no statistics for the rest of the launch period labels. Not retrieved, replenished or selected by category.','',
        '| domain | only number | Benign | Attack | Cover window | Support state |','|---|---:|---:|---:|---:|---|']
    for r in support: lines.append(f'| {r["domain"]} | 400 | {r["benign"]} | {r["attack"]} | {r["windows_covered"]} | {r["support_status"]} |')
    for d in DOMAINS:
        lines += ['',f'**{d} ** for each window:'+ '、'.join(f'{r["rank_start"]}–{r["rank_end_exclusive"]}: {r["sample_count"]}' for r in windowcounts if r['domain']==d)+ '。',
            '** primitive category counts (including Benign) **:'+'、'.join(f'{r["original_attack_category"]}: {r["count"]}' for r in categorycounts if r['domain']==d)+'。']
    lines += ['', '## follow-up range and actual waiting time','',
        'Keeping the original window boundary, excluding window ID and old tail; Full start-up period 100000 items are excluded (including unlabeled 99600 items). P7 identity of the rank／original_row／record_id／availability_ms／window_id of the 64560000 clause; The primitive line index is a complete sequence and corresponds to the end time and the primitive line stable sequence, so the calibration identity and identity values intersect as 0 in each domain. The window was not rebuilt.','',
        '| range | Subsequent rank range | items | complete windows | Old tail excludes | start Wait for second |','|---|---|---:|---:|---:|---:|']
    for r in scope: lines.append(f'| {r["domain"]} | [100000,{r["evaluation_rank_end_exclusive"]}) | {r["evaluation_rows"]} | {r["evaluation_complete_windows"]} | {r["old_excluded_tail"]} | {r["startup_wait_seconds"]} |')
    lines += ['', '** follow-up aggregate: 64260000 items, 6426 as complete windows.**',
        'The waiting time is defined as the same field as rank 99999 minus FLOW_END_MILLISECONDS minus rank 0; It is the time interval between the start of a sample by an existing stream-ending availability agent, containing the intervals in the data, without any artificial marking, calculation or on-line latency. The original start time has been saved and verified from CSV in phases, and availability_ms is consistent with the original end time. The original CSV is non-native record_id and manifest as indicated; domain:original_row using an existing P7 and not a fictitious native identity. original row indices starts with 0, excluding the CSV head. Border and time batch situation by zone see evaluation_scope.csv, do not move boundaries.','',
        '## Follow-up Asset and Information Boundary','',
        'All six source base-model artifacts, training manifests, and six frozen Platt calibrators are present; 18 structures NPY simultaneously provide Raw/Platt and 36 domains × models × output conditions. Verify the number of file bytes, NPY shape/dtype, domain/model, features sequence and availability rank index contracts. This time, no reading of probability numerals or running reasoning; The historical total probability limitation and mapping verification comes from A/B1/B2 and the current availability conclusions are limited to assets and metadata.',
        '18 The finite coefficients of All49 f(u) JSON, positive scale, convergence state, fold/model/Raw-Platt/target/development, are all in line with fit_manifest and freeze. The formula for 1-u and f(u) and the original B2 float64 CUDA arithmetic input have been verified and can be subsequently calculated by corresponding probabilities without weighting the fit; This time, the actual score was not calculated.',
        'C0-select two development domain R_select complete windows range, original tail with corresponding probability and labels caching metadata; It is a new control to be built, this time with no a_dev option. Its equal weighting domain risk is (e1+e2)/(c1+c2), not the direct average of the two risk domains. The scope and budget are as follows:','',
        '| Fold | Target | development domain | R_select range | items | excludes tail |','|---|---|---|---|---:|---:|']
    for r in devscope: lines.append(f'| {r["fold"]} | {r["target"]} | {r["domain"]} | [{r["rank_start"]},{r["rank_end_exclusive"]}) | {r["rows"]} | {r["excluded_tail"]} |')
    lines += ['', 'This round’s target 400 clause labels is only for later method-shared threshold choices, and does not enter detector, Platt or f(u) recruitment. detector /Platt from UNSW; f(u) is derived from the double development domain R_fit. Similarly recorded in history are other folding developmental roles, retaining this existing role replication and clearly not constituting an additional independent sample.',
        f' Upward {len(before)} size/mtime correspondence before and after a file; The subtype text asset is an alternate nucleus SHA256, with the model hashed to asset_availability.csv. All new files are written in this preview directory only, the old model, configuration, report, freeze results remain the same.',
        'The historical abstract has written the sum of P7 as 65560000, and A has been interpreted as a narrative plus sum error; The current window is actually 64560000, this time by subtracting 300000 from 64260000. This historical text difference does not change the freeze window and does not cover the old documentation.','',
        '## audit materials','',
        'The main deliveries are C_preflight_report.md, calibration_sample_manifest.csv, calibration_support.csv, evaluation_scope.csv, asset_availability.csv, protocol_draft.json, validation.json.',
        'Additionally: calibration_ranks.csv, sampling sequence /, rank, NPY, rank_commitment.json, calibration_window_counts.csv, calibration_attack_category_counts.csv, development_R_select_scope.csv, source_evidence.csv, label_access_audit.json, before and after upstream_inventory, run_preflight.py, commands.sh, run.log. The draft agreement is still pending review, and the C1/C2 rules have been written and have not been implemented.','',
        '## in preparation for conclusion by domain','']
    for r in support:
        ok=r['label_support_sufficient']
        lines.append(f'- **{r["domain"]}**: labels supported '+('That’s enough.' if ok else 'It’s not enough (TARGET_CLASS_SUPPORT_INSUFFICIENT)')+f'，Benign={r["benign"]}、Attack={r["attack"]}; Full assets; The agreement did not find any conflict with the data scope.'+('Fixed samples, identity isolation, evaluation range and six model Raw/Platt assets have the prerequisite conditions for entering C1 and are subject to authorization after user verification.' if ok else 'identity isolation, scope of assessment and asset conditions but do not meet the C1 support thresholds for each 20 category; New deployable thresholds cannot be generated according to the scheme, and the domain and the C0 can be controlled without changing the sample. It’s a lack of labels support, not a model performance failure.'))
    lines += ['', 'C-preflight stops at this point, waiting for user verification and then enters C1.']
    (OUT/'C_preflight_report.md').write_text('\n'.join(lines)+'\n')
    log('C-preflight complete; stopped before C1/C2.')

if __name__=='__main__':
    try: main()
    except Exception as e:
        write('failure.json',dict(time_utc=now(),error=repr(e),rank_resampling_allowed=False))
        raise
