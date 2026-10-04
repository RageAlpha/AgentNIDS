from common import *
from collections import Counter

BMET=['coverage','effective_attack_detection','wrong_accepted_attack_fraction','selective_risk']
def n(v): return None if v=='NA' else float(v)
def same(a,b):
    if a is None or b is None: assert a is None and b is None
    else: assert abs(float(a)-float(b))<1e-12
def pct(v): return 'NA' if v is None or v=='NA' else f'{float(v)*100:.4f}'
def table(headers,data): return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in data]
def boot(c,bg):
    def ratio(a,b): return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)
    c=c.astype(float); bg=bg.astype(float)
    return dict(coverage=ratio(c.sum(axis=1),bg.sum(axis=1)),effective_attack_detection=ratio(c[:,3],bg[:,2:].sum(axis=1)),
        wrong_accepted_attack_fraction=ratio(c[:,2],bg[:,2:].sum(axis=1)),selective_risk=ratio(c[:,1]+c[:,2],c.sum(axis=1)))

def main():
    check_frozen(); assert read(OUT/'independent_candidate_validation.json')['status']=='PASS'
    metrics=rows(OUT/'selective_metrics.csv'); temporal=rows(OUT/'temporal_metrics.csv'); cats=rows(OUT/'attack_category_metrics.csv'); tc=rows(OUT/'temporal_attack_category_metrics.csv')
    pairs=rows(OUT/'paired_comparisons.csv'); cal=rows(OUT/'target_calibration_metrics.csv'); dev=rows(OUT/'development_selection_metrics.csv')
    status=rows(OUT/'configuration_status.csv'); c2=rows(OUT/'C2_acceptance_thresholds.csv'); c0=rows(OUT/'C0_select_thresholds.csv')
    lookup={(r['domain'],r['model'],r['output'],r['score'],r['method']):r for r in metrics}
    assert len(lookup)==len(metrics)==108 and len(temporal)==432 and len(pairs)==144
    finalchecks=[]; verdicts=[]
    for r in status:
        d,m,o,s=[r[v] for v in ['domain','model','output','score']]; k=dict(**key(d,m,o),score=s)
        with np.load(OUT/f'window_aggregates/{d}__{m}__{o}__{s}.npz') as w:
            wc=w['counts']; avail=w['method_available']; assert float(w['classification_threshold'])==.5
            assert int(w['evaluation_rank_start'])==100000 and list(w['confusion_order'])==COUNTS
        assert avail.tolist()==[True,d!='CIC',True]
        assert (wc[2].sum(axis=(1,2))==10000).all()
        bgtotal=wc[2].sum(axis=(0,1)); totals=(bgtotal[:2].sum(),bgtotal[2:].sum()); bounds=np.r_[0,np.cumsum(SEGMENTS[d])]
        for mi,method in enumerate(['C0-select','C2-select','C0-class']):
            row=lookup[d,m,o,s,method]; count=wc[mi].sum(axis=(0,1)) if avail[mi] else None
            v=selective(count,totals)
            for name,value in v.items(): same(n(row[name]),value)
            if avail[mi]:
                assert int(row['accepted_rows'])==int(row['benign_accepted'])+int(row['attack_accepted'])
                assert int(row['accepted_errors'])==int(row['FP'])+int(row['FN'])
                if int(row['accepted_rows'])==0: assert row['selective_risk']=='NA'
                if row['status']==NOFEAS: assert int(row['accepted_rows'])==0 and float(row['threshold'])==2.0
            else: assert row['threshold']=='NA' and row['accepted_rows']=='NA' and row['status']==UNSUPPORTED
            ct=[x for x in cats if all(x[a]==k[a] for a in k) and x['method']==method]
            assert sum(int(x['category_support']) for x in ct)==int(row['rows'])
            for x in ct:
                ci=int(x['original_attack_code']); sup=int(wc[2,:,ci].sum()); assert sup==int(x['category_support'])
                vals=selective(wc[mi,:,ci].sum(axis=0) if avail[mi] else None,(sup,0) if x['original_attack_category']=='Benign' else (0,sup))
                for a,b in vals.items(): same(n(x[a]),b)
            tr=sorted([x for x in temporal if all(x[a]==k[a] for a in k) and x['method']==method],key=lambda x:int(x['segment']))
            assert [int(x['windows']) for x in tr]==SEGMENTS[d]
            for si,x in enumerate(tr):
                a,b=bounds[si:si+2]; bg=wc[2,a:b].sum(axis=(0,1)); c=wc[mi,a:b].sum(axis=(0,1)) if avail[mi] else None
                vals=selective(c,(bg[:2].sum(),bg[2:].sum()))
                for name,value in vals.items(): same(n(x[name]),value)
                catseg=[z for z in tc if all(z[a]==k[a] for a in k) and z['method']==method and int(z['segment'])==si+1]
                assert sum(int(z['category_support']) for z in catseg)==int(x['rows'])
                if avail[mi]:
                    assert all(sum(int(z[v]) for z in catseg)==int(x[v]) for v in COUNTS)
                else: assert all(z[v]=='NA' for z in catseg for v in COUNTS)
            if avail[mi]: assert all(sum(int(x[v]) for x in tr)==int(row[v]) for v in COUNTS)
            if mi<2:
                good=avail[mi] and int(row['accepted_rows'])>0 and 20*int(row['accepted_errors'])<=int(row['accepted_rows']) and 10*int(row['accepted_rows'])>=int(row['rows'])
                verdicts.append(dict(**k,method=method,status=row['status'],empirical_risk_coverage_joint_pass=bool(good) if avail[mi] else None,
                    coverage=n(row['coverage']),risk=n(row['selective_risk']),effective_attack_detection=n(row['effective_attack_detection']),
                    deployment_guarantee=False))
        mult=np.load(C1/f'bootstrap/{d}_block_multiplicity.npy'); bg=mult@np.add.reduceat(wc[2].sum(axis=1),np.arange(0,DOMAINS[d][2],10))
        rates=[]
        for mi in range(2): rates.append(boot(mult@np.add.reduceat(wc[mi].sum(axis=1),np.arange(0,DOMAINS[d][2],10)),bg) if avail[mi] else {v:np.full(2000,np.nan) for v in BMET})
        with np.load(OUT/f'bootstrap_replicates/{d}__{m}__{o}__{s}.npz') as br:
            for v in BMET:
                delta=rates[1][v]-rates[0][v]
                assert np.array_equal(delta,br['delta_'+v],equal_nan=True)
                assert np.array_equal(rates[0][v],br['C0_'+v],equal_nan=True)
                assert np.array_equal(rates[1][v],br['C2_'+v],equal_nan=True)
                q=next(q for q in pairs if all(q[a]==k[a] for a in k) and q['metric']==v)
                a=n(lookup[d,m,o,s,'C0-select'][v]); b=n(lookup[d,m,o,s,'C2-select'][v])
                same(n(q['difference']),b-a if a is not None and b is not None else None)
                good=delta[np.isfinite(delta)]; assert int(q['valid_replicates'])==len(good)
                if len(good):
                    ci=np.percentile(good,[2.5,97.5],method='linear'); same(n(q['ci_low']),ci[0]); same(n(q['ci_high']),ci[1])
                else: assert q['ci_low']==q['ci_high']=='NA'
        finalchecks.append(dict(**k,all_accept_matches_C1=True,counts_and_class_support_consistent=True,
            category_temporal_window_aggregation=True,bootstrap_replicates_and_intervals=True,unsupported_and_full_reject_distinct=True))
    # Add the actual C0 development-selection -> target-evaluation gap, separately from target400 diagnostics.
    gap=[x for x in rows(OUT/'calibration_evaluation_gap.csv') if x['comparison_source']!='development_equal_weight_vs_target_evaluation']
    for x in dev:
        if x['development_domain']!='two_domains': continue
        k={v:x[v] for v in ['domain','fold','model','output','score']}; e=lookup[x['domain'],x['model'],x['output'],x['score'],'C0-select']
        for metric in ['coverage','selective_risk']:
            a=n(x[metric]); b=n(e[metric])
            gap.append(dict(**k,method='C0-select',selection_source='complete development R_select',comparison_source='development_equal_weight_vs_target_evaluation',
                metric=metric,calibration_value=a,evaluation_value=b,evaluation_minus_calibration=b-a if a is not None and b is not None else None))
    save(OUT/'calibration_evaluation_gap.csv',gap); save(OUT/'conclusion_checks.csv',verdicts)
    check_frozen()
    before=read(OUT/'upstream_inventory_before.json'); after=inventory(); assert before==after
    write(OUT/'upstream_inventory_after.json',after)
    assert all(sha(a['path'])==a['sha256'] for a in read(OUT/'score_definition_and_assets.json')['assets'])
    assert read(OUT/'evaluation_access.json')['started_at_utc']>read(OUT/'threshold_freeze.json')['frozen_at_utc']
    val=dict(status='PASS_C2_C0_SELECT_COMPLETE_STOPPED',completed_at_utc=now(),configurations=36,C0_selected=sum(x['status']=='SELECTED' for x in c0),
        C0_full_reject=sum(x['status']==NOFEAS for x in c0),C2_selected=sum(x['status']=='SELECTED' for x in c2),C2_full_reject=sum(x['status']==NOFEAS for x in c2),C2_unsupported=12,
        unique_evaluation_records=64260000,evaluation_windows=6426,classification_always_half=True,C1_thresholds_used=False,
        original_rank_and_400_manifest_hashes_unchanged=True,thresholds_frozen_before_own_fold_target_evaluation=True,
        original_B1_B2_score_functions_used=True,independent_candidate_audit='independent_candidate_validation.json',checks=finalchecks,
        bootstrap='C1 matrices reused by hash-verified references; no draws; percentile linear finite replicates',
        upstream_files_unchanged=len(before),preservation='size/mtime all upstream; hashes of small text, all preflight/C1 files, detector/Platt/fu artifacts',
        empirical_joint_pass_count=sum(v['empirical_risk_coverage_joint_pass'] is True for v in verdicts),
        scripts_sha256={p.name:sha(p) for p in OUT.glob('*.py')},calibration_resampling=False,refitting=False,
        AURC_optimization=False,other_experiments_run=False)
    write(OUT/'validation.json',val)
    lines=['# C2-select and C0-select audit reports','',
        '** has completed 36 group development domain C0-select, 24 group C2-select with adequate target support, CIC group 12 with C2 with insufficient support with NA. It’s been verified. There is no actual C0/C2 follow-up operation point to reach both risk≤5% and coverage≥10%.**','',
        'C2 is only ToN / MLP-All49 / Raw / 1 − u finds calibration as a valid threshold, the rest of the 23 group is undefined and the whole abstention is missing. The only non-empty C2 followed by coverage=7.0975%, risk=9.3526%, and neither of the thresholds continued. C0-select has 10 group development available; 26 group is abstention and 10 group is not at the same time at the target. All abstention risk is NA, not a safe success; CIC support is insufficient without C2 movement, and cannot be written as a whole abstention.','',
        '## sources, protocols and information boundaries','',
        f' Project root directory `{ROOT}`;` output this time.{OUT}`. Read ` directly {PLAN}`，SHA256 `{PLAN_SHA}` corresponds to /C1 pre-examination. The original path of the annex with SHA256 is recorded in protocol.json/protocol_freeze.json.',
        f' This phase protocol freeze time `{read(OUT/"protocol.json")["frozen_at_utc"]}`; All C0/C2 thresholds freeze `{read(OUT/"threshold_freeze.json")["frozen_at_utc"]}`; The evaluation process begins with `{read(OUT/"evaluation_access.json")["started_at_utc"]}`. Selection and evaluation is a independent process, evaluating non-imported selection functions.',
        'This is a look at the A/B1/B2 after design, an exploratory experiment of target domain after unblinding, and can’t be traced back to old pre-registered or new unseen domain verification. This round ensures that the threshold is only freeze and then calculates the threshold value of the threshold. Other historical target role replications of R_select development have been disclosed, not claiming to have ever been viewed in relevant labels history.',
        'ToN 343/57, BoT 41/359, CIC 400/0 (Benign/Attack) is a direct repeat of the pre-tested fixed 400 clause. There is no sampling, no labels supplementation, no class balancing, no extended start-up or retraining. Classification of fixed p≥0.5; There is no classification threshold selected using C1. Retain all 36 configurations: three All49 models × Raw/Platt × two scores × three domains.',
        'Entropy directly calls original B2.entropy (same xlogy float64 CUDA formula as B1),and f(u) calls frozen B2.predict. Check18 JSONs and fold/model/output/development/fit_manifest/hash. Accept high-score s≥a ties as complete groups,without clipping,rounding or reversal. Float64 boundaries-1/2 give all-accept/all-abstain;save decimal/hex and confirm serialized actions unchanged.','',
        '## Fixed Scope and Development Budget','',
        'The entire [0,100000) start-up period is excluded, including undrawn 99600 items. The following ToN [100000,27520000), 2742 windows; CIC [100000,20110000), 2001 window; BoT [100000,16930000), 1683 windows, total 64260000 items /6426 windows. calibration / evaluates identity without intersection, excluding identity and tail along the original window. The ToN/CIC boundary still belongs to the same available time batch and does not claim strict clock separation.','']
    lines+=table(['fold','Objectives','development domain','R_select range','Record count'],[[r['fold'],r['target'],r['domain'],f'[{r["rank_start"]},{r["rank_end_exclusive"]})',r['rows']] for r in rows(PRE/'development_R_select_scope.csv')])
    lines+=['','C0 equally weights two development domains: coverage=(A1/N1+A2/N2)/2 and risk=(E1/N1+E2/N2)/(A1/N1+A2/N2). It’s the same with CA=A1×N2+A2×N1 and CE=E1×N2+E2×N1. It’s bound to 5×CA≥N1×N2, 20×CE≤CA. Check that 40×N1×N2 is below the upper limit of int64. independent audits use approximate integers and Python Fraction, without mixing the two risk domains directly or by record number.',
        'C0 is compared to C2 by the label source and actual budget, not the equivalent labels budget causal experiment. The development_selection_metrics.csv subdomain provides the original counting with the divider, the coverage/risk using the equal weighting subdomain, and the combined line accepting the number as the original total, without pretending to be the equal weighting ratio divider.','',
        '## candidate coverage with freeze status','',
        'The full C2 candidate table preserves the 400 point devalue and boundary, each point being marked with both the / error / and the acceptable and optional values. C0 lacks the precise combinations of the two development domain scores, without using the approximate grid: save the total number of candidates, all available successive group spaces, select full statistics with the score for each 65536 group / cumulative number SHA256.',
        'Each non-empty rating group in the development option increases the CA rigorously, so that the last feasible group maximizes the coverage only; The highest score thresholds prevail when all acceptable boundaries are counted against the lowest scoring group. independent reviews with an alternative unique/inverse plus-group rectangle diagram, re-calculates all candidate abstract hashes per block, and then checks for all possible groups with the best possible results using constraints; It’s not just about checking out a selection.','']
    lines+=table(['domain','Models','The output','Scores','C0 state','C2 state'],[[r['domain'],r['model'],r['output'],r['score'],r['C0_status'],r['C2_status']] for r in status])
    lines+=['','## follow-up operation points and weights','',
        'The following table retains all 36 configurations. coverage/risk as a percentage; NA and 0 are strictly distinguished. All real execution conditions are fixed at classification threshold 0.5. The no-abstention C0-class context is recomputed in selective_metrics.csv. Its accept-all confusion counts match the C1-stage C0-class counts on the same scope item by item.','']
    data=[]
    for r in status:
        d,m,o,s=[r[v] for v in ['domain','model','output','score']]; a=lookup[d,m,o,s,'C0-select']; b=lookup[d,m,o,s,'C2-select']
        data.append([d,m,o,s,pct(a['coverage']),pct(a['selective_risk']),pct(b['coverage']),pct(b['selective_risk'])])
    lines+=table(['domain','Models','The output','Scores','C0 cov%','C0 risk%','C2 cov%','C2 risk%'],data)
    lines+=['','**Did target labels change operating points?** Among 24 configurations with sufficient support, 7 have thresholds/actions differing from C0: two ToN XGBoost configurations change from nonempty acceptance to full abstention, one ToN MLP changes from full abstention to nonempty acceptance, and four BoT RF configurations change to full abstention; The other two methods for the 17 group are all abstention. A change in thresholds is not the same as a performance improvement.',
        '** is the only non-empty C2: ** ToN MLP-All49 Raw 1 − u accepts 90 in 400 clause ((22.5%), incorrect 4 clause ((4.4444%), accepts 86 clause Benign and 4 clause Attack , 4 clause Attack all misjudge Benign and correctly accept the attack as 0. Subsequently, 1946125 records are accepted: acceptance falls to 7.0975% and risk rises to 9.3526%. Its C0 contrasts with the full abstention and risk are almost undefined. It can only be compared with the coverage and the attack results of the non-empty action with the entire abstention and cannot be claimed to have reduced C0 risk.',
        'The two types of acceptance rate and attack results for all subsequent non-empty operation points are %:','']
    nonempty=[r for r in metrics if r['method']!='C0-class' and r['accepted_rows'] not in ['0','NA']]
    lines+=table(['domain','Model / output / rating','The method','Benign acceptance rate','Attack acceptance rate','effective attack detected','Wrong acceptance of Attack','Attack abstention'],[
        [r['domain'],f'{r["model"]}/{r["output"]}/{r["score"]}',r['method']]+[pct(r[v]) for v in ['benign_acceptance','attack_acceptance','effective_attack_detection','wrong_accepted_attack_fraction','attack_rejection']] for r in nonempty])
    lines+=['','** low risk with the category: ** CIC RF 1 − u C0 risk is only 2.5897% but coverage=7.3117% is lower than 10%, correctly accepts Attack as 0 and low risk cannot be considered a successful detection. CIC RF’s f(u) can be more accurately attacked, but produces about 76.9% to total risk. ToN XGBoost C0 approximately 39.68% accepts, 16.20% risk accepts very few attacks correctly; BoT RF C0 only accepts about 0.0584%, 92.32% risk. The only nonempty C2 action mainly accepts Benign; effective detection among all Attack records is only about 0.0510%.',
        'selective_metrics.csv includes TN/FP/FN/TP for the full accepted set, FPR among accepted Benign, false-negative rate among accepted Attack, original attack counts, and every denominator. It is not possible to test the accuracy of an internal Attack evaluation. calibration_evaluation_gap.csv simultaneously marks the C2 target calibration → rating, C0 develops equal weighting selection→ rating and C0 the diagnostic gap on the target 400, not mixed sources.','',
        '## time stability and original category','',
        'There are four windows along C1: ToN 686/686/685/685, CIC 501/500/500/500 and BoT 421/421/421/420; All paragraphs have the same threshold. The only non-empty C2 fractional result:','']
    lines+=table(['Section 1','Total number','Acceptance of numbers','coverage%','risk%','effective Attack%','Wrong acceptance of Attack%'],[
        [r['segment'],r['rows'],r['accepted_rows']]+[pct(r[v]) for v in ['coverage','selective_risk','effective_attack_detection','wrong_accepted_attack_fraction']]
        for r in temporal if r['domain']=='ToN' and r['model']=='mlp_All49' and r['output']=='Raw' and r['score']=='one_minus_u' and r['method']=='C2-select'])
    lines+=['','The following non-empty operation points of the primary category are highlighted; effective detection and error acceptance at all abstention points in other executions are 0 and abstention rates at 100% when supported in this class; The C2 of CIC is still NA, not following the full abstention rule.','']
    focus={'ToN':['scanning'],'BoT':['Reconnaissance'],'CIC':['FTP-BruteForce','DoS_attacks-SlowHTTPTest']}
    lines+=table(['domain','Categories','Model / output / rating','The method','Support','Acceptance rate %','effective detects %','Wrong acceptance of %','abstention %'],[
        [r['domain'],r['original_attack_category'],f'{r["model"]}/{r["output"]}/{r["score"]}',r['method'],r['category_support']]+[pct(r[v]) for v in ['coverage','effective_attack_detection','wrong_accepted_attack_fraction','attack_rejection']]
        for r in cats if r['method']!='C0-class' and r['original_attack_category'] in focus[r['domain']] and lookup[r['domain'],r['model'],r['output'],r['score'],r['method']]['accepted_rows'] not in ['0','NA']])
    lines+=['','All categories maintain their original taxonomy; BoT Theft excludes tail from the original, with zero support for NA. All categories are accepted at different times, correct / error acceptance and abstention split into attack_category_metrics.csv and temporal_attack_category_metrics.csv, with / grouped by / grouped by window.','',
        '## coupled with statistical and interpretational limitations','',
        'Repeat the block index of the new C1 rating range with multiplicity, domain-by-domain hashes, complete block ranges, complete verification of window order; 275/201/169 blocks, each 10 window, the last short block retained, 2000 times. All configurations and methods in the same domain are shared, no new matrix is generated, and no old P7 matrix is read with a start-up period. bootstrap_sources.csv preserves the absolute source and hash.',
        'C2 − C0’s coverage, effective attack detection, incorrect acceptance of the attack ratio report the matching difference between 95% percentile and linear fractional range, using the real class divider after resampling per copy; Undefined duplicate excludes and reports effective numbers. All 36 groups do not have two methods for simultaneous non-empty acceptance, so selective risk is less than all NA and cannot falsify a risk improvement interval. CIC is all C2 minus NA.','']
    lines+=table(['Indicators','Difference pp','95% range pp','effective copy'],[
        [r['metric'],pct(r['difference']),f'[{pct(r["ci_low"])},{pct(r["ci_high"])}]',r['valid_replicates']]
        for r in pairs if r['domain']=='ToN' and r['model']=='mlp_All49' and r['output']=='Raw' and r['score']=='one_minus_u'])
    lines+=['','Operating points with different acceptance rates compared to the above ranges are not compared to coverage. Conditional on a single 400 sampling, the freeze model and the current already unblinding domain, does not cover calibration sampling uncertainty, and does not provide risk assurance for the future domain. AURC changes are not used as an indicator of improvement at this stage.','',
        '## testing and stopping','',
        f' validation.json:PASS. independent candidate grouping, precise domain equal weighting, sequential boundaries, original scoring direction, first freeze post-evaluation, fully accepted background, two types of / primitive category / time period / by window counting, full abstention with insufficient support, matrix and interval matching are passed. Upward {len(before)} The size/mtime of a file remains unchanged; Pre-check /C1 The entire file, model /Platt/f(u) and small text hash remain unchanged.',
        'This directory preserves protocol.json/protocol_freeze.json, score_definition_and_assets.json, configuration_status.csv, complete C2 candidates and thresholds, C0 candidate abstracts and audit materials, threshold_freeze.json, development of / targets calibration / follow-up / category / time indicators, gaps / pair tables, bootstrap sources and copies, necessary window aggregation and verification. Candidate audit is stored in aggregate abstract and whole set hashes, without copying the full probability caching or new incremental trajectory.',
        'The script: common.py, prepare.py, select_acceptance.py, evaluate.py, verify_candidates.py, report_and_validate.py; The execution order is accompanied by a log. The independent candidate review only reviews existing decisions after evaluation, without re-selecting or rewriting the freeze threshold.',
        '** stops when this phase is completed. No joint threshold search, K sensitivity, new CRE, resistivity, coordination, assembly or other roadmap experiments have been performed.**']
    (OUT/'stage_C2_report.md').write_text('\n'.join(lines)+'\n')
    log(f'Final validation PASS. {len(before)} upstream files unchanged. No target C0/C2 point jointly passes risk/coverage. Stopped.')

if __name__=='__main__': main()
