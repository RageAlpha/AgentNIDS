"""Validate small B1 artifacts and build descriptive Chinese reporting only."""
from pathlib import Path
import csv
import json
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
A=ROOT/'outputs/post_p7_diagnostics/stage_A/run_20260914_A1'
MODELS=['rf_Traffic','rf_All49','xgb_Protocol','xgb_All49','mlp_Behavior','mlp_All49']
DOMAINS=['UNSW','ToN','CIC','BoT']
PCTS=[.01,.05,.1,.2,.4,.6,.8,1.]
def read(name): return list(csv.DictReader((OUT/name).open()))
def save(name,rs):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pc(x,digits=3): return 'NA' if x=='NA' else f'{100*float(x):.{digits}f}'
def dec(x): return f'{float(x):.3e}' if abs(float(x))<1e-5 else f'{float(x):.6f}'

def main():
    overall=read('overall_selective_metrics.csv');ops=read('operating_points.csv');cats=read('attack_category_selective_metrics.csv')
    common=read('common_coverage_expected_risks.csv');regions=read('diagnostic_regions.csv');scopes=read('evaluation_scope.csv')
    look={(r['domain'],r['model'],r['output']):r for r in overall}
    opmap={(r['domain'],r['model'],r['output'],float(r['requested_coverage'])):r for r in ops}
    comap={(r['domain'],r['model'],r['output'],float(r['requested_coverage'])):r for r in common}
    assert len(overall)==48 and len(ops)==384 and len(common)==384
    stagea=list(csv.DictReader((A/'detector_raw_platt_metrics.csv').open()))
    amap={(r['domain'],r['model'],r['output']):r for r in stagea}
    for r in ops:
        count=int(r['accepted_count']);bn=int(r['benign_accepted']);an=int(r['attack_accepted'])
        assert count==bn+an==sum(int(r[k]) for k in ['TN','FP','FN','TP'])
        assert int(r['accepted_errors'])==int(r['FP'])+int(r['FN'])
        assert count<=int(r['capacity_count'])
        selected=[c for c in cats if (c['domain'],c['model'],c['output'],c['requested_coverage'])==(r['domain'],r['model'],r['output'],r['requested_coverage'])]
        assert sum(int(c['category_total']) for c in selected)==int(r['attack_total'])
        assert sum(int(c['accepted_count']) for c in selected)==an
        assert sum(int(c['correct_accepted_attack_count']) for c in selected)==int(r['TP'])
        assert sum(int(c['wrong_accepted_benign_count']) for c in selected)==int(r['FN'])
        assert sum(int(c['abstained_count']) for c in selected)==int(r['attack_abstained'])
        if float(r['requested_coverage'])==1:
            ar=amap[r['domain'],r['model'],r['output']]
            assert [int(r[k]) for k in ['TN','FP','FN','TP']]==[int(ar[k]) for k in ['TN','FP','FN','TP']]
    for c in cats:
        assert int(c['category_total'])==int(c['correct_accepted_attack_count'])+int(c['wrong_accepted_benign_count'])+int(c['abstained_count'])
        if int(c['category_total'])==0:
            assert all(c[k]=='NA' for k in ['category_coverage','effective_category_detection','wrong_accepted_fraction','abstained_fraction'])
    curves=list((OUT/'risk_coverage_curves').glob('*.npz'))
    assert len(curves)==48
    for p in curves:
        d,m,o=p.stem.split('__');z=np.load(p)
        assert len(z['expected_risk'])==1000
        assert np.all(np.diff(z['expected_prefix_count'])>=0)
        assert np.all(np.diff(z['deterministic_count'])>=0)
        assert np.all(z['deterministic_count']<=z['expected_prefix_count'])
        assert np.isfinite(z['expected_risk']).all()
        full=float(look[d,m,o]['full_accept_error_rate'])
        assert abs(z['expected_risk'][-1]-full)<1e-15 and abs(z['deterministic_risk'][-1]-full)<1e-15

    # Same-coverage differences use expected tie-order curves only. Deterministic
    # risks are not compared as if their requested coverages were achieved exactly.
    diffs=[]
    for d in DOMAINS:
        for pct in PCTS:
            pairs=[]
            for split in ['rf_Traffic','xgb_Protocol','mlp_Behavior']:
                full=split.split('_')[0]+'_All49'
                for o in ['Raw','Platt']: pairs.append(('All49_minus_split',split,o,full,o))
            for m in MODELS: pairs.append(('Platt_minus_Raw',m,'Raw',m,'Platt'))
            for effect,m1,o1,m2,o2 in pairs:
                x=comap[d,m1,o1,pct];y=comap[d,m2,o2,pct]
                assert x['prefix_count']==y['prefix_count']
                diffs.append(dict(domain=d,effect=effect,requested_coverage=pct,common_coverage=x['common_coverage'],
                    from_model=m1,from_output=o1,to_model=m2,to_output=o2,from_expected_risk=x['expected_risk'],
                    to_expected_risk=y['expected_risk'],expected_risk_difference_pp=100*(float(y['expected_risk'])-float(x['expected_risk'])),
                    interpretation='same-coverage expected within-tie risk; not deterministic action',improvement_direction='negative'))
    save('same_coverage_risk_differences.csv',diffs)
    focuspairs=[('ToN','scanning'),('BoT','Reconnaissance'),('CIC','FTP-BruteForce'),('CIC','DoS_attacks-SlowHTTPTest')]
    focus=[r for r in cats if (r['domain'],r['original_attack_category']) in focuspairs]
    save('focus_attack_categories.csv',focus)
    ovrows=[]
    for d in DOMAINS:
        for m in MODELS:
            r=look[d,m,'Raw'];p=look[d,m,'Platt']
            ovrows.append([d,m,dec(r['AURC']),dec(p['AURC']),pc(r['full_accept_error_rate'])+'→'+pc(p['full_accept_error_rate'])])
    ovtable=table(['domain','Models','Raw AURC↓','Platt AURC↓','100% acceptance error rate % Raw → Platt'],ovrows)
    op40=[]
    for d in ['ToN','CIC','BoT']:
        for m in ['rf_All49','xgb_All49','mlp_All49']:
            for o in ['Raw','Platt']:
                r=opmap[d,m,o,.4]
                op40.append([d,m,o]+[pc(r[k]) for k in ['actual_coverage','selective_risk','benign_coverage','attack_coverage','effective_attack_detection','wrong_accepted_attack_fraction']])
    optable=table(['domain','All49 model','The output','Acceptance of %','Total risk %','Benign to %','Attack to %','effective detects %','False acceptance of the % attack'],op40)
    targetregions=[r for r in regions if r['domain']!='UNSW']
    regtable=table(['domain','Models','The output','Acceptance endpoint range %','risk endpoint %','effective attack detects the endpoint of %','End-to-end misrepresentation of the %','FPR% at the end of Benign'],[
        [r['domain'],r['model'],r['output'],pc(r['start_coverage'])+'–'+pc(r['end_coverage']),pc(r['start_risk'])+'→'+pc(r['end_risk']),
            pc(r['start_effective_attack_detection'])+'→'+pc(r['end_effective_attack_detection']),pc(r['end_wrong_accepted_attack_fraction']),pc(r['end_accepted_benign_fpr'])] for r in targetregions])
    focusrows=[]
    for d,c in focuspairs:
        for m in ['rf_All49','xgb_All49','mlp_All49']:
            r=next(z for z in focus if z['domain']==d and z['original_attack_category']==c and z['model']==m and z['output']=='Raw' and float(z['requested_coverage'])==.4)
            p=next(z for z in focus if z['domain']==d and z['original_attack_category']==c and z['model']==m and z['output']=='Platt' and float(z['requested_coverage'])==.4)
            focusrows.append([d+'/'+c,m,f"{int(r['category_total']):,}"]+[pc(r[k])+'→'+pc(p[k]) for k in ['category_coverage','effective_category_detection','wrong_accepted_fraction','abstained_fraction']])
    focustable=table(['Domain/original category','All49 model','Total number of categories','Acceptance of % in class','effective detects %','It’s wrong for Benign%.','abstention %'],focusrows)
    scope_table=table(['domain','Partition','Record count','Benign','Attack','Excluding tail'],[[s['domain'],s['partition']]+[f"{int(s[k]):,}" for k in ['rows','benign','attack','excluded_tail']] for s in scopes])
    nempty=sum(r['accepted_count']=='0' for r in ops)
    validation=json.loads((OUT/'validation.json').read_text())
    assert len(list((OUT/'figures').glob('*.png')))==8
    validation.update(status='PASS_B1_COMPLETE',delivery_checks=dict(operating_points=384,attack_category_rows=len(cats),
        empty_operating_points=nempty,zero_support_rows=sum(int(r['category_total'])==0 for r in cats),
        category_balance_from_csv=True,stage_A_confusion_from_csv=True,curve_final_risk=True,
        compressed_curve_files=48,curve_display_grid=1000,same_coverage_comparisons=len(diffs),figures=8,
        figures_visually_reviewed=(OUT/'figure_review.json').exists()),stage_B2_started=False,stage_C_started=False)
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')

    scope_doc=f'''# B1 scope, rating and indicator definition

This directory: `{OUT}`. A stage Source: `{A}`. stage_A_report.md, asset_and_scope.md, evaluation_scope.csv, detector_raw_platt_metrics.csv, and class tables and verification records have been read. B1 is an exploratory diagnosis based on the same batches of unblinding data that have already been collected without producing new independent validation.

## Data and asset reuse

{scope_table}

UNSW uses S_source_test rank `[2130000,2365424)` and contains the end 5,424. ToN / CIC / BoT uses the P7 F1 / F3 / F2 T_heldout complete 10k window forewings, initializing the window retains; Each excludes the 260 / 5,529 / 3,808 clause. The goal is to summarize the 64,560,000 line, the 6,456 window. No other development folded domain records have been added together.

P3 cache path `outputs/prediction_cache_v2/<domain>/<model>.npy`; Raw reads raw_probability and Platt reads probability. The six models rf_Traffic, rf_All49, xgb_Protocol, xgb_All49, mlp_Behavior, and mlp_All49 are all retained. All49 is the main candidate, dividing the view into opposites, and not selecting the winner according to the target results.

0=Benign / 1=Attack tested along A phase; model with classes_ / probability column/ calibration apparatus identity; native class encoded. labels comes from `outputs/feature_store_v2/*_label.npy`, the original category comes from `*_attack_code.npy` and attack_codebook in manifest domains. Recordings are arranged according to availability rank and the corresponding original line is determined by the original sorted_original_rows.npy. B1 the complete windows boundary, endpoints, front-drive and P7 identity ledger rows of the same partition and checked_windows nucleus; Phase A is directly replicated for the full record_id/original_row audit, without claiming B1 or reading the entire identity string in step by step.

source domain also continues to use the P3 probability of phase A; The numerical difference between P2/P3 and MLP is still effective. At this stage, no model reasoning has been completed, and there is no fit Platt, f(u) or detector.

## Fixed Classification and Score

`yhat = 1[p >= 0.5]`; The probability is exactly the same as 0.5 for Attack.

`u = -(xlogy(p,p) + xlogy(1-p,1-p))/ln(2)` , `s=1-u` , s from high to low. float64 and cupyx.scipy.special.xlogy are used to define 0 · ln(0)=0, so p=0/1 is u=0, s=1 and p=0.5 is u=1, s=0.

No p increases the epsilon interruption, no u/s is inserted, tightened or reversed; The exact ties group that maintains the actual float64 score. The original freeze Platt contained an input interrupt with its existing output properties, which has not been replicated or reversed at this stage. Repeated probability, symmetrical conjugation, original model conjugation, and original Platt intersections and floating point expressions can all form ties; Not broken down by real category or original row indices.

Raw scores only evaluate Raw classification accuracy and Platt scores only Platt classification accuracy. The order and acceptance moves read s only; The real labels is only used for statistics after the same universal acceptance set is generated.

## AURC rules, curves and operating points of ties

In the `src/nids/evaluation/round1_metrics.py`, directly replicate ranking_groups, grouped_aurc, harmonic_numbers. AURC represents the arithmetic mean of all N non-aircraft precursors risk, randomly ranking the expected ** for ** in each precise homologation; It is not just the average point of the module, it is not the gradient point of the map point.

If a group accepts n0 and false e0, the group accepts m and false e. The expectation of a t in the group is that risk is `(e0+t·e/m)/(n0+t)`. This analysis expects non-realistic sampling and does not allow labels to determine the order within the group. The AURC and P7 of the nine target domain main view Platt have verified that the one_minus_u values differ from the 1e-12 values.

The fixed operating points are 1%, 5%, 10%, 20%, 40%, 60%, 80%, 100%. For the requested proportion q, the capacity K=floor(qN), select the largest complete group precursor to the cumulative group number ≤K. If the threshold score is t, the acceptance condition is s≥t; The first group accepted the set as empty when it was above K, the actual acceptance rate 0, risk, NA. Capacity used hundreds of points multiplied without additional loss through floating point errors. The next group must be overcapacity, and the set has been accepted with q loosening nesting.

In the graph, the linear curves in the group in the random order ** expect ** to be used to compare the same acceptance rate; A circular point uses a definitive whole-set operation point, where the horizontal coordinate is the actual acceptance rate, which may be significantly different from the requested value. The point on the expected curve located within the large ties group is not an executable deterministic operation point and cannot be used as deployment evidence. If there is no acceptable forecast, do not draw the risk circle, do not draw the risk as zero.

`risk_coverage_curves/` compresses one NPZ for each condition, preserving the 1,000 display points calculated from the total quantity sequence (requesting 0.1% to 100%) and the actual number of individuals/risk for the corresponding whole group progenitor. It is a display sample of the curve, not a complete step-by-step trajectory or complete lossless set output; AURC and diagnostic area checks still cover all accurate arrays, and are not related to the display grid. Curve array fields are shown in the directory README.md.

`common_coverage_expected_risks.csv` provides an expected risk under the same number of prospects with eight specified ratios;`same_coverage_risk_differences.csv`’s All49 − split view, Platt − Raw deviation unit is a percentage point, lowered to negative risk. The actual acceptance rates for different model determination operation points are different and do not imply the same acceptance rate difference.

## operational point divider

- total selective risk =(FP+FN)/(TN+FP+FN+TP) and TN/FP/FN/TP have been recorded.
- Benign acceptance rate =(TN+FP)/ all Benign; Benign false positive rate =FP/(TN+FP) has been accepted.
- Attack acceptance rate =(FN+TP)/ all Attack; Attack false negative rate =FN/(FN+TP) has been accepted.
- ** effective attack detects =TP/ all Attack**; False acceptance rate of attack =FN/ all Attack; Attack abstention for example =( all Attack − TP − FN)/ all Attack and all three for 1.
The correct acceptance and incorrect acceptance of the - primitive attack category, abstention, are all subtracted by the ** record number ** and not replaced by the acceptance number; The acceptance rate of a category is the sum of its acceptance numbers /.

The total zero acceptance of risk is NA; When a real class is not recorded, the class accepts the condition error rate NA, but if the class as a whole is empty, its effective detects =0, abstention and =1. All the proportions of the zero support categories are NA and the counting retains 0. BoT Theft is supported as 0 after excluding P7 tail along the unchanged category set. The overall ratio is 0 to 1; In Chinese, the report is shown as a percentage; AURC is the original scale.

## 5% / 10% area for diagnosis only

`diagnostic_regions.csv` is the sum of all satisfied 20× is the cumulative error number ≤ is the cumulative acceptance number 10× is the cumulative acceptance number ≥N is the cumulative acceptance number ** is the complete port point **. The actual acceptance rate, risk, the two types of acceptance rate, the detection of effective attack, the error acceptance attack and the Benign condition false positive rate at each of the adjacent qualified terminal points. There may be unavailable coverage gaps between interval endpoints caused by ties, and it is not possible to say that the acceptance rate of each real number within the range is applicable.

The table does not select a threshold or a model and no threshold is exported to deployment; The first end only describes the entire regional boundary. All source conditions exist at these endpoints; target domain area details see report. This is described after the unblinding data, not as a risk assurance, no new bootstrap or significant conclusions.

## calculation and inspection

Read from the original cache block on the 262,144 line, calculating only one condition each time. Calculation, sorting, grouping, AURC using the existing RTX A5000 CUDA process; CPU writes only the polymer CSV/ compression array and uses existing matplotlib diagrams. There is no replication of all predicted caches or output of new incremental trajectories. Main computing time {validation['elapsed_seconds']:.2f} Seconds; For version, endpoint and condition-by-condition checks see validation.json.

The 100% confusion count for all 48 conditions has been verified as equivalent to the A phase, two types of decomposition of total acceptance numbers/error numbers, summation of all primitive attack categories, nested acceptance range and endpoints, maximum group foreplay, nine historical Platt AURC correspondence values. The new output is unchanged in this running directory, the old model, configuration, A and P7 phases files.
'''
    (OUT/'scope_and_score_definition.md').write_text(scope_doc)
    (OUT/'risk_coverage_curves'/'README.md').write_text('''# curve data

Display sampling points on each `<domain>__<model>__<Raw|Platt>.npz` with 1,000 in full quantity order; There is no identity or prediction.

- requested_coverage:0.001…1.
- expected_prefix_count:floor(N×i/1000); expected_coverage: The number is /N.
- expected_error , expected_risk: the exact number of expected errors in a homogeneous random sequence within the ties group, expected risk.
- deterministic_count , deterministic_error: not more than the maximum number of forecasts for the whole group.
- deterministic_coverage , deterministic_risk: actual whole group acceptance rate and risk; zero acceptance at risk NaN

The curve shows the sample is not a sampling evaluation data; The indicator uses this condition to record everything. AURC is calculated using an existing formula for all non-pre-pre-existing coefficients, without scoring a point on this 1,000. All 5%/10% diagnostic areas come from all achievable endpoints, not display points. The curves and operating points have different ties portfolios, see upper scope_and_score_definition.md.
''')

    report=f'''# B1: selective-prediction diagnosis with fixed entropy scores

** has completed the 48 condition, 384 fixed operating point, validation passed.** All49 is the main candidate, with the split view retained as a counterpoint. Results that do not support the hypothesis that rejection of high hypothesis records generally reduces the hypothesis error rate: some models have higher priority area errors; In some low overall risk regions, almost no effective attacks were detected. The following is a post-diagnosis of the unblinding data, not a new independent verification, deployment threshold selection or risk assurance. B1 stops after completion.

## 1. scope and scores

{scope_table}

with A as recorded: the source test retains 5,424 as tail; The goal is to take only complete windows, with a total of 64,560,000 items, excluding 9,597 items tail, without repeating the calculation of the development breakdown. Raw/Platt is read separately from the P3 cache; All are classified according to p≥0.5 and classified according to s=1− from high to low acceptance, with no additional probability of interception, reversal of scores or fit.

The middle line of the AURC/ graph expects ** in random order along ** within the same group; The fixed point of operation takes no more than the maximum ** whole group ** for the specified acceptance rate, and the round point shows the actual acceptance rate. The two are not to be confused with the same action that can be implemented. all {nempty} A zero acceptance operating point risk is NA; The category is still not supported with NA. Definition, file fields and resource description see [scope_and_score_definition.md](scope_and_score_definition.md).

## 2. Six models by domain AURC with total acceptance error rate

{ovtable}

The lower the AURC, the better, but it is not possible to determine an attack detecting effective using only AURC or total risk. The whole value: [overall_selective_metrics.csv](overall_selective_metrics.csv). Expectations of the same acceptance rate for risk and the corresponding differential value for [common_coverage_expected_risks.csv](common_coverage_expected_risks.csv) and [same_coverage_risk_differences.csv](same_coverage_risk_differences.csv) do not directly compare the operating points of different actual acceptance rates as the same coverage.

![UNSW risk Acceptance rate ](figures/UNSW_risk_coverage.png)

![ToN risk Acceptance rate ](figures/ToN_risk_coverage.png)

![CIC risk Acceptance rate ](figures/CIC_risk_coverage.png)

![BoT risk Acceptance rate ](figures/BoT_risk_coverage.png)

Raw divided by Platt, six model graphs are consistent; All49 real line, split view of the null line. The line is the 1,000 point of the expected curve, and the circle points are the operating points of the whole group. The 5%/10% reference line is not a deployment rule. The source longitudinal axis is 0, the cylinder is 3.1%, the target graph is 0, the cylinder is 102%, and the two sides of the same field are the same size. Additional four areas of effective attack detected the chart at figures/ and the subdivision is always all real Attack.

## 3. Fixed Operating Point: View both acceptance and attack at the same time

The following table uniquely shows all target domain All49 candidates with ** requests for 40% acceptance rate ** without selecting the model or operating point according to the result; Eight operating points, six complete models are kept at [operating_points.csv](operating_points.csv). The risk fraction in this table is the accepted total, and the remainder of the class proportions are recorded using all real classes.

{optable}

Effective detection = correctly accepted Attack/all Attack; incorrectly accepted attacks = incorrectly accepted Benign/all Attack. Together with attack abstention they sum to100%; recall among accepted attacks is not a substitute.

A decrease in the acceptance rate of ** is not the same as a decrease in risk.** ToN XGBoost-All49 Raw total acceptance error rate 14.397%, actual 9.970% , risk 23.618% when requested by 10%, and worse when rejected. BoT XGBoost-All49 Raw Full acceptance error rate 18.565% , actual 9.986% , risk 98.102% when requesting 10%; RF-All49 is the same as the actual 9.980%, risk 91.966%. These results directly show that the priority ratings can be focused on errors. Overall, source domain is good, and CIC’s RF-All49/XGBoost does have a low risk subset. risk is not universally monotonous with change in acceptance rate.

** low risk may obscure the leakage.** CIC XGBoost-Protocol Platt when requesting 40% the actual 39.981% , risk 1.752% , but Benign acceptance rate 45.128% , Attack acceptance rate 5.330%; The correct acceptance of an attack is only 1,023, which accounts for the entire Attack **0.0393%**, the incorrect acceptance of Benign as 137,611 (5.291%) and the attack on abstention 94.670%.**99.262%** is still a leak in the Attack accepted; Accept the set 98.276% as Benign and the original total Benign as 87.067%. The low risk in this region, with a large acceptance of Benign, excludes Attack, cannot be called a effective attack detection.

The majority of ** effects are not entirely Benign.** BoT is the prime total 99.696% is Attack. When RF-Traffic Platt requests 80%, the actual 76.961%, total risk 0.700%, effective attack detects 76.640%, incorrectly accepts 0.478% attack, and more correct Attack is accepted; However, the Benign acceptance rate is only 25.256%, and the Benign has received FPR **81.509%**. In general, good does not represent both. The above breakdown is an observation, not a category-specific statement of the reasons for failure.

**Large tie groups matter.** CIC MLP-All49 Platt first ties cover50.631%; requested1%–40% accepts0. ToN first ties cover23.251%; requested1%–20% accepts0. BoT MLP-Behavior Platt requested40% realizes only2.597%,risk0.259%,effective attack detection2.598%,with all accepted Benign falsely alerted. This is not low risk at40% acceptance. Source RF-All49 Platt first ties cover77.196%;extremely low AURC does not make arbitrary small acceptance rates attainable.

## 4. Focused Attack Category

Complete calculation of the original attack category, acceptance, correct detection, misconnection and abstention see [attack_category_selective_metrics.csv](attack_category_selective_metrics.csv), combined {len(cats)} Going; All operating points in the four focus categories are separated by [focus_attack_categories.csv](focus_attack_categories.csv). The category set and the original spelling remain unchanged, BoT Theft supports 0, all proportions NA.

The following table still fixes ** to 40%** request, with each graph being **Raw → Platt**; The percentages are divided by the total number of records in the original category. The corresponding actual overall acceptance rate is shown in the table above, in particular CIC MLP-All49 Platt 0 acceptance.

{focustable}

- **ToN scanning : When ** requests 40%, the three All49 detections of effective are less than 0.64%, mistaking Benign for about 29.57%, 18.46% and 36.94%–37.19% respectively; It’s not a safe abstention for the whole class. Raw XGBoost-All49 receives a lower rating of about 0.9876, MLP-All49 about 0.99948 and is a high-rating error. effective detects 0.494%→14.948% at the same point as RF-Traffic, but its overall risk is high and cannot pick the winner according to a category.
- **BoT Reconnaissance : ** Raw XGBoost-All49 When requesting 10%, the lower limit is about 0.9709, which has been mistakenly labeled **81.942%** as Benign and effective as 0. When requesting 40%, RF/XGBoost/MLP-All49 is incorrectly attached to this type about 91.65%/93.50%/88.60%; MLP Raw has a lower score of about 0.999993. The 40% of the RF distribution view is less prone to point error, but about 95% the class is detected by abstention, with effective detecting only 0.364%.
- **CIC FTP-BruteForce , DoS_attacks-SlowHTTPTest: ** XGBoost-All49’s Raw and Platt both accept the **100% error as Benign** (386,720 / 105,550), with Raw scoring about 0.958 below the 40% when requested. MLP-All49 Raw is the same as 100%, and the lower bound is about 0.99999936; Platt This is a global acceptance of 0, not a correction of the prediction. The Platt is completely wrong when it is accepted. RF-All49 , XGBoost-Protocol at the request point are all abstention in both categories, and the effective detected is still 0; Although RF-Traffic detects both at full A acceptance, the priority acceptance point is also full abstention.

## 5. Region with risk≤5% and acceptance≥10%

source domain 12 conditions are all regions from which complete module point checks can be carried out;**ToN All conditions do not include ** in this region. The goal is to produce only the following 8 conditions:All models are retained and no deployment thresholds are selected according to this table.

{regtable}

This is only an endpoint segment that meets the descriptive conditions on the unblinding record, and there may be unrealized acceptance rate gaps within the segment; It’s not a continuous coverage guarantee. All segments with two endpoints are statistically defined. See [diagnostic_regions.csv](diagnostic_regions.csv). effective attacks in these regions of CIC are very low: RF-All49 is at most about 0.083%, Protocol is at most about 0.039% and XGBoost-All49 is about 0; So the low total risk cannot determine success. BoT RF-Traffic effective attacks detected higher, but the end Benign condition misreporting still needs to be single-column, and not cross-domain assurance.

## 6. Conclusions and boundaries

1. Does risk decrease when the receptivity of ** decreases?** source domain is generally lower than risk; CIC RF-All49 / XGBoost has a low risk subset; None of the three ToN All49 models has a 5%/10% operating region; some BoT All49 models perform particularly poorly in their highest-priority region.
2. ** High confidence error is still present.** ToN scanning , BoT Reconnaissance , CIC XGBoost / MLP errors of the above two types of attacks can be accepted when s approaches 1. He also said that the government would not automatically eliminate the lack of confidence.
3. ** is predominantly accepted only in the majority of classes. Some areas with low risk are obviously so, but cannot be generalized.** CIC The low risk zone is oriented towards Benign and the effective attack detects close to 0; BoT RF is biased towards most Attack and detects a large number of attacks to mask Benign misrepresentations. The category composition, condition error rate and effective detection should be read together.
Is 4. **All49 more suitable for selective prediction? There is no consistent cross-domain conclusion.** ToN The AURC of the three-structure All49 is lower than the split view, but the effective attack detection remains poor; CIC RF-All49 is better, XGBoost-All49 is more Protocol, MLP Raw is less Behavior, and Platt AURC is better and there is a large acceptance gap; BoT RF-All49 is substantially worse than Traffic. XGBoost-All49 improves over Protocol but retains confident false negatives; MLP-All49 also has worse AURC. target domain winners were not selected.
5. **Raw/Platt is domain dependent.** RF-Traffic AURC deteriorates significantly in ToN/CIC and improves significantly in BoT; MLP-All49 in CIC improves, ToN deteriorates. Platt can change the ranking, classification results and uncertainty on both sides of the ties group; At this stage, the respective contributions of these mechanisms are not causal decomposed.
6. ** cannot determine the future effect or the risk guarantee.** No changes in classification threshold, direction or model at this stage, no new significant conclusions/bootstrap, no Strict-K, preservation, coordinated experiments or follow-up stages. What B2/C directions are worth continuing and what independent data is needed, awaiting a decision after review.

Verification: The number of 100% operation points confused by 48 is completely consistent with A phase; Acceptance and error decomposition, summation of the original attack categories, zero support for NA, acceptance of the entire nested set, endpoints and ranges are passed; Nine historical Platt AURC correspondence values are consistent. Record identity audits reused according to asset statements, without repeating claims of new independent validation. The evidence is [validation.json](validation.json).

Delivery directory: `{OUT}`. The script is executed: run_b1.py, plot_b1.py, report_b1.py; Orders and logs see [commands.sh](commands.sh), [run.log](run.log), [plot.log](plot.log), [report.log](report.log). All old models, configurations and reports remain unchanged.**B1 has been completed and has not entered B2 or C.**
'''
    (OUT/'stage_B1_report.md').write_text(report)
    print(json.dumps(validation['delivery_checks'],ensure_ascii=False,indent=2))

if __name__=='__main__': main()
