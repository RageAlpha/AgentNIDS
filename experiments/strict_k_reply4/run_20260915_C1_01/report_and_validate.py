"""Phase 3: audit saved sufficient statistics and produce review materials.
This phase does not select or alter thresholds and does not read evaluation labels.
"""
from common import *
from collections import Counter

def number(v): return None if v=='NA' else float(v)
def equal(a,b):
    if a is None or b is None: assert a is None and b is None
    else: assert abs(float(a)-float(b))<1e-12
def pct(v): return 'NA' if v is None or v=='NA' else f'{float(v)*100:.3f}'
def mdtable(headers,data):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in data]

def main():
    check_frozen()
    thresholds=rows(OUT/'classification_thresholds.csv'); candidate=rows(OUT/'classification_candidates.csv')
    metricsrows=rows(OUT/'classification_metrics.csv'); temporal=rows(OUT/'temporal_metrics.csv')
    categories=rows(OUT/'attack_category_metrics.csv'); tcats=rows(OUT/'temporal_attack_category_metrics.csv')
    pairs=rows(OUT/'paired_comparisons.csv'); gaps=rows(OUT/'calibration_evaluation_gap.csv')
    sample=rows(PRE/'calibration_sample_manifest.csv')
    protocol=read(OUT/'protocol.json')
    lookup={(r['domain'],r['model'],r['output'],r['method'],r['split']):r for r in metricsrows}
    assert len(lookup)==len(metricsrows)==144
    assert len(temporal)==288 and len(thresholds)==36
    checks=[]; conclusions=[]; completed=[]
    for r in thresholds:
        d,m,o=r['domain'],r['model'],r['output']; k=key(d,m,o)
        active=r['status']!=UNSUPPORTED
        subset=[x for x in candidate if (x['domain'],x['model'],x['output'])==(d,m,o)]
        rec=[x for x in sample if x['domain']==d]
        p=np.array(cache(d,m)[FIELDS[o]][[int(x['availability_rank']) for x in rec]],dtype=np.float64)
        y=np.array([int(x['binary_label']) for x in rec],dtype=np.uint8)
        if active:
            wanted=np.unique(np.r_[p,0.,.5,np.nextafter(np.float64(1),np.float64(np.inf))])
            assert np.array_equal(wanted,[float(x['threshold']) for x in subset])
            for x in subset:
                t=np.float64(x['threshold']); assert float(t).hex()==x['threshold_hex']
                # Recheck candidate membership and direct decisions against saved counts.
                c=confusion(y,p,t)
                assert list(c)==[int(x[v]) for v in COUNTS]
                assert (x['feasible']=='True')==(20*int(c[1])<=int((y==0).sum()))
                equal(number(x['FPR']),div(int(c[1]),int((y==0).sum())))
                equal(number(x['Recall']),div(int(c[3]),int((y==1).sum())))
            feasible=[x for x in subset if x['feasible']=='True']
            optimum=min(feasible,key=lambda x:(-int(x['TP']),int(x['FP']),abs(float(x['threshold'])-.5),-float(x['threshold'])))
            chosen=[x for x in subset if x['selected']=='True']
            assert len(chosen)==1 and chosen[0]==optimum and optimum['threshold']==r['threshold']
            assert sum(int(x['TN'])+int(x['FN'])==400 for x in subset)>=1
            assert float(subset[-1]['threshold'])>1
            half=next(x for x in subset if float(x['threshold'])==.5)
            c0=lookup[d,m,o,'C0-class','calibration']
            assert all(int(half[v])==int(c0[v]) for v in COUNTS)
            if r['status']==DEGENERATE: assert int(optimum['TP'])==0
        else:
            assert d=='CIC' and not subset and r['threshold']=='NA'
        with np.load(OUT/f'window_aggregates/{d}__{m}__{o}.npz') as w:
            counts=w['counts']; assert list(w['confusion_order'])==COUNTS
            assert list(w['methods'])==(['C0-class','C1-class'] if active else ['C0-class'])
            assert counts.shape[1]==DOMAINS[d][2] and (counts.sum(axis=(2,3))==10000).all()
        bounds=np.r_[0,np.cumsum(SEGMENTS[d])]
        for mi,method in enumerate(['C0-class','C1-class']):
            present=mi==0 or active
            for split in ['calibration','evaluation']:
                row=lookup[d,m,o,method,split]
                if not present:
                    assert all(row[v]=='NA' for v in METRICS+['threshold','accepted_rows'])
                else:
                    c=np.array([int(row[v]) for v in COUNTS])
                    assert c.sum()==int(row['rows'])==int(row['accepted_rows'])
                    assert (c[:2].sum(),c[2:].sum())==(int(row['benign']),int(row['attack']))
                    expected=metrics(c)
                    for v in METRICS: equal(number(row[v]),expected[v])
                    if split=='evaluation': assert np.array_equal(counts[mi].sum(axis=(0,1)),c)
                cr=[x for x in categories if (x['domain'],x['model'],x['output'],x['method'],x['split'])==(d,m,o,method,split)]
                assert sum(int(x['category_support']) for x in cr)==int(row['rows'])
                if present:
                    assert all(sum(int(x[v]) for x in cr)==int(row[v]) for v in COUNTS)
                    for x in cr:
                        if not int(x['category_support']): assert all(x[v]=='NA' for v in ['attack_recall','benign_FPR','category_correct_rate'])
                else: assert all(x[v]=='NA' for x in cr for v in COUNTS+['attack_recall','benign_FPR','category_correct_rate'])
            tr=sorted([x for x in temporal if (x['domain'],x['model'],x['output'],x['method'])==(d,m,o,method)],key=lambda x:int(x['segment']))
            assert [int(x['windows']) for x in tr]==SEGMENTS[d]
            assert sum(int(x['rows']) for x in tr)==DOMAINS[d][1]-100000
            if present:
                assert all(sum(int(x[v]) for x in tr)==int(lookup[d,m,o,method,'evaluation'][v]) for v in COUNTS)
                for si,x in enumerate(tr):
                    assert np.array_equal(counts[mi,bounds[si]:bounds[si+1]].sum(axis=(0,1)),[int(x[v]) for v in COUNTS])
                    catseg=[z for z in tcats if (z['domain'],z['model'],z['output'],z['method'],int(z['segment']))==(d,m,o,method,si+1)]
                    assert all(sum(int(z[v]) for z in catseg)==int(x[v]) for v in COUNTS)
                    equal(float(x['threshold']),.5 if mi==0 else float(r['threshold']))
            else: assert all(x[v]=='NA' for x in tr for v in METRICS+['threshold','accepted_rows'])
        mult=np.load(OUT/f'bootstrap/{d}_block_multiplicity.npy')
        sampled=np.load(OUT/f'bootstrap/{d}_sampled_blocks.npy')
        assert mult.shape==sampled.shape==(2000,(DOMAINS[d][2]+9)//10)
        assert (mult.sum(axis=1)==mult.shape[1]).all()
        for rep in [0,999,1999]: assert np.array_equal(np.bincount(sampled[rep],minlength=mult.shape[1]),mult[rep])
        with np.load(OUT/f'bootstrap/{d}__{m}__{o}_replicates.npz') as b:
            for mi,method in enumerate(['C0-class','C1-class']):
                if mi==1 and not active: continue
                bc=mult@np.add.reduceat(counts[mi].sum(axis=1),np.arange(0,DOMAINS[d][2],10))
                rates=rate_arrays(bc)
                for j,v in enumerate(COUNTS): assert np.array_equal(b[f'C{mi}_{v}'],bc[:,j])
                for v in RATES: assert np.array_equal(b[f'C{mi}_{v}'],rates[v],equal_nan=True)
            for v in METRICS:
                delta=b[f'C1_{v}']-b[f'C0_{v}']
                assert np.array_equal(delta,b[f'delta_{v}'],equal_nan=True)
                q=next(x for x in pairs if (x['domain'],x['model'],x['output'],x['split'],x['metric'])==(d,m,o,'evaluation',v))
                good=delta[np.isfinite(delta)]
                assert int(q['valid_replicates'])==len(good)
                if len(good):
                    ci=np.percentile(good,[2.5,97.5],method='linear')
                    equal(number(q['ci_low']),ci[0]); equal(number(q['ci_high']),ci[1])
                else: assert all(q[x]=='NA' for x in ['C1','difference','ci_low','ci_high'])
        for q in [x for x in pairs if (x['domain'],x['model'],x['output'])==(d,m,o)]:
            v=q['metric']; x=number(lookup[d,m,o,'C0-class',q['split']][v]); z=number(lookup[d,m,o,'C1-class',q['split']][v])
            equal(number(q['difference']),z-x if x is not None and z is not None else None)
        for q in [x for x in gaps if (x['domain'],x['model'],x['output'])==(d,m,o)]:
            v=q['metric']; x=number(lookup[d,m,o,q['method'],'calibration'][v]); z=number(lookup[d,m,o,q['method'],'evaluation'][v])
            equal(number(q['evaluation_minus_calibration']),z-x if x is not None and z is not None else None)
        a=lookup[d,m,o,'C0-class','evaluation']; b=lookup[d,m,o,'C1-class','evaluation']
        pass_fpr=(20*int(b['FP'])<=int(b['benign'])) if active else None
        recall_up=(int(b['TP'])>int(a['TP'])) if active else None
        fpr_down=(int(b['FP'])<int(a['FP'])) if active else None
        recall_down=(int(b['TP'])<int(a['TP'])) if active else None
        label='Inadequate support, C1 not implemented' if not active else 'Benign decay is not a success' if r['status']==DEGENERATE else (
            'FPR Experience Score and Recall Improved' if pass_fpr and recall_up else 'FPR experience standard, but decrease in error report accompanied by decrease in Recall' if pass_fpr and recall_down else
            'FPR not reached; A decrease in error report accompanied by a decrease in Recall' if not pass_fpr and fpr_down and recall_down else
            'FPR not reached; Recall improvement accompanied by an increase in misreports' if not pass_fpr and recall_up else 'FPR not reached; Equivalent to C0' if all(a[v]==b[v] for v in COUNTS) else 'Failure to meet two conditions for experience improvement')
        conclusions.append(dict(**k,status=r['status'],C1_evaluation_FPR_le_5_percent=pass_fpr,
            C1_recall_above_C0=recall_up,FPR_decreased=fpr_down,Recall_decreased=recall_down,
            interpretation=label,future_domain_risk_guarantee=False))
        checks.append(dict(**k,candidate_coverage_and_lexicographic_selection_verified=True if active else None,
            sentinel_float64_correct=True if active else None,full_acceptance_and_confusion_consistent=True,
            window_category_temporal_aggregates_consistent=True,bootstrap_saved_matrices_and_intervals_verified=True,
            unsupported_C1_not_fabricated=True,calibration_gap_and_paired_differences_verified=True))
        completed.append(dict(**k,C0_class_status='COMPLETE',C1_class_status=r['status'],
            C1_evaluation_status='COMPLETE' if active else 'NOT_RUN_SUPPORT_INSUFFICIENT',C2_status='NOT_RUN',C0_select_status='NOT_RUN'))
    save(OUT/'conclusion_checks.csv',conclusions); save(OUT/'configuration_completion.csv',completed)
    check_frozen()
    draft=read(PRE/'protocol_draft.json')
    assert protocol['C2']==draft['C2'] and protocol['C0_select']==draft['C0_select']
    before=read(OUT/'upstream_inventory_before.json'); after=inventory()
    assert before==after
    write(OUT/'upstream_inventory_after.json',after)
    freeze=read(OUT/'protocol_freeze.json')
    assert all(sha(OUT/n)==h for n,h in freeze['script_sha256'].items())
    access=read(OUT/'evaluation_access.json')
    assert access['first_evaluation_label_access_after_utc']>read(OUT/'threshold_freeze.json')['frozen_at_utc']
    validation=dict(status='PASS_C1_COMPLETE_STOPPED_FOR_REVIEW',completed_at_utc=now(),
        plan_sha256=PLAN_SHA,protocol_frozen=True,thresholds_frozen_before_evaluation_labels=True,
        selected_C1=24,unsupported_C1=12,degenerate_C1=sum(r['status']==DEGENERATE for r in thresholds),
        actual_C0_conditions=36,actual_C1_conditions=24,classification_metric_rows=len(metricsrows),
        evaluation_records_unique=64260000,evaluation_windows_unique=6426,
        fixed_400_sample_manifest_and_rank_commitments_unchanged=True,calibration_resampling_performed=False,
        identity_disjointness='identity_validation.json: current original-row scan intersection=0; unchanged full P7 preflight identity ledger',
        candidate_checks=checks,bootstrap_replicates=2000,bootstrap_scope='fresh shifted 10-window blocks, final short block retained, per-domain shared matrices',
        bootstrap_effective_replicate_counts=dict(Counter(x['valid_replicates'] for x in pairs if x['split']=='evaluation')),
        undefined_rates='NA with effective replicate count; no zero imputation',
        old_files_unchanged=len(before),preservation='all upstream size/mtime; <=2MB text hashes; every preflight file SHA256; original execution-script hashes preserved',
        C2_rules_unchanged=True,C2_executed=False,C0_select_executed=False,detector_Platt_fu_retraining=False,
        AUROC_AP_Brier_ECE_used_for_improvement=False,report_script_sha256=sha(Path(__file__)),
        runtime=dict(python=sys.version,numpy=np.__version__,computation='CPU NumPy; int64 counts and float64 probabilities, thresholds and metric ratios'),
        interpretation='already unblinded exploratory domains; empirical FPR attainment is not a future-domain guarantee')
    write(OUT/'validation.json',validation)
    lines=['# Strict-K C1-class and C0-class audit reports','',
        '** has completed 36 group C0-class and 24 group C1-class; The 12 group C1 of CIC maintains insufficient support for NA. Verification is approved, stop waiting for the audit.**',
        'Two groups of XGBoost-Protocol Raw/Platt of ToN reaching FPR≤5% and Recall above C0 in the following range; ToN The rest of the 10 group is a complete decomposition of Benign. BoT is only MLP-Behavior Raw/Platt with two groups reaching FPR≤5%, but the Recall decreases. These results cannot be generalized to a general improvement in cross-domain.','',
        '## protocol, freeze and data boundaries','',
        f' Project root directory: `{ROOT}`. This is the list: `{OUT}`。',
        f' Plans: `{PLAN}`; SHA256 ` Designed by 2026-09-14 {PLAN_SHA}` is consistent with the pre-examination of the audit.',
        f' independent protocol.json (strict-k-reply4-C1-v1.0) is produced by pre-testing protocol_draft.json and freeze by `.{protocol["frozen_at_utc"]}`. All the freeze values are at `.{read(OUT/"threshold_freeze.json")["frozen_at_utc"]}`; labels visits recorded at `{access["first_evaluation_label_access_after_utc"]}`. Selection and evaluation is done by the independent process, which only loads the saved thresholds.',
        'The restriction of only preflight is maintained in protocol.json’s history_preflight. This stage authorizes C1, C0-class, subsequent classification evaluation and bootstrap; C2/C0-select has not been implemented and the rules are the same as the draft.',
        'This exploratory design follows inspection of A/B1/B2. All three target domains are already unblinded; this is not new unseen-domain validation. All methods directly share each domain’s original 400 stations, without retrieving, replenishing labels, balancing categories, or extending the launch period; Untrained detector, Platt or f(u).',
        'The entire [0,100000) start-up period is excluded, including unlabeled 99600 items. Using the existing full 10k window, the original ID window and the old tail rules. The two boundary records of ToN/CIC have the same availability_ms; identity is non-intersecting, but does not claim strict clock time separation.','']
    scope=rows(OUT/'evaluation_scope.csv')
    lines+=mdtable(['domain','calibration Benign/Attack','Evaluation of distances','Record count','complete windows','The number of windows','Old tail excluded'],[
        [d,f'{DOMAINS[d][3]}/{DOMAINS[d][4]}',f'[100000,{DOMAINS[d][1]})',DOMAINS[d][1]-100000,DOMAINS[d][2],'/'.join(map(str,SEGMENTS[d])),next(x['old_excluded_tail'] for x in scope if x['domain']==d)] for d in DOMAINS])
    lines+=['','Evaluation totals **64260000 records and 6426 windows**, counting each domain once without multiplying by models or methods. 400 promises labels and pre-examination rank to remain unchanged; The current original line identity scan confirmed once again that calibration crossed with the evaluation as 0.',
        '## calibration Selection and support insufficient','',
        'Candidate for 400 is the decreasing value of the float64 probability, plus 0, 0.5, 1.0000000000000002. The final entry is nextafter(1,+ ∞ ) of float64, preserving the decimal system and hex, which is still strictly greater than 1 after reading. Classification of p≥t as an equal-probability whole group treatment; All candidates are consistent with direct comparison counts and cover all achievable calibration allocations.',
        'The possible conditions are 20×FP≤Benign numbers, ToN maximum 17 times FP and BoT maximum 2 times. The maximum TP, the lowest FP, the lowest |t − 0.5|, the highest t; The same domain-fixed divisor optimizes the first two equals the precise ratio. t=0.5 contrasts with maintaining the candidate table and the calibration indicator.','']
    lines+=mdtable(['domain','Models','The output','t_K','calibration FP/Benign','calibration TP/Attack','Status'],[
        [r['domain'],r['model'],r['output'],r['threshold'],f'{r["calibration_FP"]}/{r["calibration_benign"]}',f'{r["calibration_TP"]}/{r["calibration_attack"]}',r['status']] for r in thresholds if r['domain']!='CIC'])
    lines+=['','All10 ToN zero-recall solutions actually select a>1 sentinel and evaluate it: accept all,predict all Benign,FPR=0,Recall=0. This is neither abstention nor success;0.5 was not silently restored. ToN calibration attacks are scanning only;BoT attacks are Reconnaissance only. Support in400 records does not cover every attack class.',
        'The calibration set of CIC is 400 Benign / 0 Attack and the 12 group is TARGET_CLASS_SUPPORT_INSUFFICIENT; There is no candidate search or new classification threshold. calibration and the rated C1 mix count, rate, and difference value are NA and the sample number is kept to the range of metadata; C0 is calculated correctly. CIC calibration C0’s Recall/BA is not Attack and is marked as NA.','',
        '## followed by C1 − C0 classification results','',
        'The following table shows the percentage of FPR/Recall as a new range; First, see if C1 FPR is ≤5% and then see if Recall is improved. BA/Accuracy does not serve as a basis for the conversion of conclusions. All TN/FP/FN/TP, BA/Accuracy and calibration results see classification_metrics.csv.','']
    table=[]
    for r,c in zip(thresholds,conclusions):
        d,m,o=r['domain'],r['model'],r['output']
        if d=='CIC': continue
        a=lookup[d,m,o,'C0-class','evaluation']; b=lookup[d,m,o,'C1-class','evaluation']
        table.append([d,m,o,f'{pct(a["FPR"])}→{pct(b["FPR"])}',f'{pct(a["Recall"])}→{pct(b["Recall"])}',c['interpretation']])
    lines+=mdtable(['domain','Models','The output','FPR% C0→C1','Recall% C0→C1','Judging'],table)
    lines+=['','**ToN:** Only the two XGBoost-Protocol Raw/Platt configurations meet both empirical criteria; subsequent FPR is about 1.321% and Recall about 32.807%; The error rate is higher than C0, but still lower than 5%. It does not eliminate the remaining leaks and cannot be extended to other domains based on this domain. The rest of the 10 group decreases to FPR with Recall decreasing to 0.',
        '**BoT: ** RF-Traffic Raw’s increase in Recall is accompanied by an increase in false reports, while FPR is still about 24.920%; The Platt is recalled for the same time as most other models for reporting errors. MLP-Behavior and FPR are about the 2.996% threshold, but Recall is only about 2.935%, below C0. XGBoost-Protocol Platt 0.5 is the result of an established ties rule, whose subsequent indicator is exactly the same as C0, with FPR still higher than 5%.','',
        '**CIC C0-class (C1 all NA): **','']
    lines+=mdtable(['Models','The output','FPR%','Recall%','BA%','Accuracy%'],[
        [m,o]+[pct(lookup['CIC',m,o,'C0-class','evaluation'][v]) for v in RATES] for m in MODELS for o in FIELDS])
    lines+=['','## matches block bootstrap with gap','',
        'In the new rating window sequence, each consecutive 10 constitutes a block of complete windows, with the last short remaining. ToN/CIC/BoT is divided into 275/201/169 blocks and 20000/10000/30000 blocks. Each 2000 domain has a backdrawn block of B, and each model, Raw/Platt, C0/C1, uses the same matrix for the domain. I don’t read the old P7 bootstrap matrix.',
        'This round pre-recorded NumPy PCG64 and ToN/CIC/BoT seeds in the protocol, respectively 20260915/20260916/20260917. This only generates a resampling rating, not a re-extraction of calibration labels. 95% percentile arranged along the interval to take the 2.5/97.5 percentage with linear insert; CPU int64 block count multiplication and float64 ratio are consistent with project statistics.',
        'paired_comparisons.csv reports all confusion counts and the C1 − C0 difference in FPR, Recall, BA, Accuracy; The rate at CSV is absolutely different from 0 to 1. The following table shows the percentage difference between two groups of ToN non-deoxidized and 95% paired spaces.','']
    interval_table=[]
    for q in pairs:
        if q['domain']=='ToN' and q['model']=='xgb_Protocol' and q['split']=='evaluation' and q['metric'] in ['FPR','Recall','BA']:
            interval_table.append([q['output'],q['metric'],pct(q['difference']),f'[{pct(q["ci_low"])},{pct(q["ci_high"])}]',q['valid_replicates']])
    lines+=mdtable(['The output','Indicators','Difference pp','95% range pp','effective copy'],interval_table)
    lines+=['','All the overall indicators actually implemented and the adequate supporting coupling variance values have a 2000 to effective duplicate; CIC C1 is undefined, 0 is effective duplicate, NA interval. The general treatment of zero denominators has been achieved as NA, not filling in 0. There is no resampling for the calibration set again, the calibration difference is only descriptive.',
        'The range is conditioned on this time 400 clause calibration sampling, freeze model and current evaluation range, not covering calibration sampling uncertainty or all future domains; Multiple comparison corrections were not made, and the cross-domain was not declared successful in a single significant interval. The FPR threshold is not equal to the future domain risk guarantee. calibration_evaluation_gap.csv preserves the method-by-method, indicator-by-indicator nickel rating - calibration nickel gap, and the coefficients are different.','',
        '## timeframe and original attack category','',
        'All four segments are continuously cut along the original complete windows and the threshold is fixed; The fractional total confusion count and rate is temporal_metrics.csv and the fractional original category is temporal_attack_category_metrics.csv. The following shows per-segment C1 FPR/Recall (%) for the two nondegenerate ToN solutions and two FPR-compliant BoT solutions. Do not adjust thresholds using segment outcomes.','']
    lines+=mdtable(['domain','Models','The output','Section 1','FPR%','Recall%'],[
        [x['domain'],x['model'],x['output'],x['segment'],pct(x['FPR']),pct(x['Recall'])] for x in temporal
        if x['method']=='C1-class' and ((x['domain']=='ToN' and x['model']=='xgb_Protocol') or (x['domain']=='BoT' and x['model']=='mlp_Behavior'))])
    lines+=['','The overall ** does not represent time stability.** ToN XGBoost-Protocol The two outputs of each FPR are less than 5%, but the Recall is approximately 0.999%, 38.545%, 70.685%, and 5.209%, respectively, and the time difference is noticeable. BoT MLP-Behavior with two output phases 1 FPR≤5% and 2–4 23.520%, 18.757% and 24.008% respectively; The total 2.996% is formed by the physical Benign numerical weight of each phase, which cannot be called a full-time peak. The actual Benign/Attack fraction of the fraction is maintained at temporal_metrics.csv.']
    focus={'ToN':['scanning'],'BoT':['Reconnaissance'],'CIC':['FTP-BruteForce','DoS_attacks-SlowHTTPTest']}
    lines+=['','the original category is not merged; The reported attack class Recall is a binomial Attack instance, not an attack family polynomial accuracy. Benign on the FPR alternate line, with zero support category marked NA. All models of the key categories, two output results:','']
    ft=[]
    for x in categories:
        if x['split']=='evaluation' and x['method']=='C0-class' and x['original_attack_category'] in focus[x['domain']]:
            z=next(z for z in categories if z['split']=='evaluation' and z['method']=='C1-class' and all(z[v]==x[v] for v in ['domain','model','output','original_attack_code']))
            ft.append([x['domain'],x['original_attack_category'],x['model'],x['output'],x['category_support'],pct(x['attack_recall']),pct(z['attack_recall'])])
    lines+=mdtable(['domain','Primary category','Models','The output','Support numbers','C0 Recall%','C1 Recall%'],ft)
    lines+=['','A key gap remains: subsequent ToN scanning Recall for XGBoost-Protocol is only about 0.953%, substantially below its aggregate Recall; BoT MLP-Behavior decreases the Recall of Reconnaissance to about 0.077%. CIC three All49 and XGBoost-Protocol, MLP-Behavior to FTP-BruteForce/SlowHTTPTest with C0 Recall remaining 0; The lack of support and lack of CIC C1 results cannot claim to address these gaps.']
    lines+=['','BoT Theft remains zero support in the original complete windows rating range, with all relevant Recall being NA; tail is not included. Category, time period, window-by-window aggregation are consistent with the overall count.',
        '## verification and delivery','',
        f' validation.json for PASS: coverage of the 24 group candidate set, direct counting, whole group ties, float64 sentry and vocabulary selection; The first freeze evaluation; calibration / rating identity does not overlap; C0/C1 is fully acceptable; Category / time period / per window is consistent with the overall; The conserved bootstrap matrix and the difference between the values are consistent. Upward {len(before)} size/mtime is the same before and after a file, small text checks SHA256, and all pre-check file hashes remain unchanged.',
        'configuration_status.csv is the threshold value of freeze when / is selected to support the state snapshot, which remains the same; The final execution state is configuration_completion.csv. Degradation and insufficient support conditions remain in the aggregate, without C0 pretending to be C1.',
        'Uncalculated AUROC, AP, probability Brier/ECE or AURC as improvement indicators for this round; No C2/C0-select, sensitivity, resistivity, coordinated or clustered experiments have been conducted.',
        'Documents: protocol.json, protocol_freeze.json, threshold_freeze.json, configuration_status.csv, configuration_completion.csv, classification_candidates.csv, classification_thresholds.csv, classification_metrics.csv, calibration_evaluation_gap.csv, attack_category_metrics.csv, temporal_metrics.csv, paired_comparisons.csv, bootstrap/, window_aggregates/, and validation.json. identity refers only to both manifest and window manifest; There is no replication of the full probability caching or preservation of the progressive prediction.',
        'Scripts and commands: common.py, select_c1.py, evaluate_c1.py, report_and_validate.py, commands.sh; Diary: selection.log, evaluation.log, report_validation.logwindow_aggregates Each file is a full-fledged aggregate of method×window×native_category×(TN,FP,FN,TP), with CIC containing only C0. bootstrap preserves the block index, multiplicity, block scope and the replication of each configuration indicator.','',
        '**C1 stops at this point and returns to review. C2 and C0-select have not yet been implemented and the established rules remain unchanged.**']
    (OUT/'stage_C1_report.md').write_text('\n'.join(lines)+'\n')
    log(f'Final artifact audits PASS. {len(candidate)} candidate rows; 36 C0 / 24 C1; upstream {len(before)} files unchanged. Stopped for review.')

if __name__=='__main__': main()
