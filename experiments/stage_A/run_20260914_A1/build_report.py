"""Assemble Stage A report from this run's computed tables; verify delivery consistency."""
from pathlib import Path
import csv
import json
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
def load(name):
    return list(csv.DictReader((OUT/name).open()))
def save(name,rs):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(x): return f'{100*float(x):.3f}'
def num(x): return f'{float(x):.6f}'

metrics=load('detector_raw_platt_metrics.csv')
cats=load('attack_category_recall.csv')
scope=load('evaluation_scope.csv')
models=list(dict.fromkeys(r['model'] for r in metrics))
lookup={(r['domain'],r['model'],r['output']):r for r in metrics}
# Retain original categories absent from complete windows (BoT Theft); never redefine taxonomy.
zero_support=[]
for s in scope:
    d=s['domain']
    book=json.loads((ROOT/f'outputs/feature_store_v2/{d}_manifest.json').read_text())['attack_codebook']
    for code,name in book.items():
        existing=[r for r in cats if r['domain']==d and r['original_category']==name]
        if existing and any(int(r['rows']) for r in existing): continue
        y=np.load(ROOT/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')
        codes=np.load(ROOT/f'outputs/feature_store_v2/{d}_attack_code.npy',mmap_mode='r')
        selected=codes==int(code)
        labels=np.unique(y[selected])
        assert len(labels)==1
        assert not np.any(selected[int(s['rank_start']):int(s['rank_end_exclusive'])])
        zero_support.append(dict(domain=d,category=name,full_domain_rows=int(selected.sum()),evaluation_rows=0))
        if existing: continue
        for m in models:
            for o in ['Raw','Platt']:
                cats.append(dict(domain=d,model=m,output=o,original_category=name,binary_label=int(labels[0]),rows=0,
                    predicted_attack=0,attack_detection_rate='NA',attack_category_recall='NA',note='NA: zero records in evaluation scope'))
save('attack_category_recall.csv',cats)
assert len(metrics)==48
for r in metrics:
    assert sum(int(r[k]) for k in ['TN','FP','FN','TP'])==int(r['rows'])
    rs=[c for c in cats if (c['domain'],c['model'],c['output'])==(r['domain'],r['model'],r['output'])]
    assert sum(int(c['rows']) for c in rs)==int(r['rows'])
    assert sum(int(c['rows']) for c in rs if str(c['binary_label'])=='1')==int(r['attack'])
    assert sum(int(c['predicted_attack']) for c in rs if str(c['binary_label'])=='1')==int(r['TP'])
    assert sum(int(c['predicted_attack']) for c in rs if str(c['binary_label'])=='0')==int(r['FP'])
for s in scope:
    for r in [r for r in metrics if r['domain']==s['domain']]:
        assert all(int(r[k])==int(s[k]) for k in ['rows','benign','attack'])
for f in load('prediction_flip_summary.csv'):
    rr=lookup[f['domain'],f['model'],'Raw'];pp=lookup[f['domain'],f['model'],'Platt']
    key='TN' if f['true_class']=='Benign' else 'TP'
    assert int(pp[key])-int(rr[key])==int(f['raw_wrong_platt_right'])-int(f['raw_right_platt_wrong'])
p3=json.loads((ROOT/'outputs/round1_v2/P3/P3_manifest.json').read_text())
for p in p3['predictions']:
    dm=json.loads((ROOT/f'outputs/prediction_cache_v2/{p["domain"]}/manifest.json').read_text())
    assert p==dm
p2audit=load('source_p2_inference_audit.csv')
assert all(float(r['delta'])==0 for r in p2audit)
history=load('historical_metric_comparison.csv')
assert all(r['match']=='True' for r in history if r['source']=='P7')
assets=load('model_asset_audit.csv')
for r in assets:
    r['native_decision']=('Raw > 0.5; equal gives Benign; code inspected, native predictions not recomputed'
        if r['model'].startswith('xgb') else ('nvforest class assignment: multi-column argmax (first class on ties), single-column > threshold; native not recomputed' if r['model'].startswith('rf') else 'Single sigmoid output; no native classifier predict method'))
save('model_asset_audit.csv',assets)
valid=json.loads((OUT/'validation.json').read_text())
final=dict(status='PASS_STAGE_A_COMPLETE',main_calculation=valid,delivery_metric_conditions=len(metrics),
    category_rows=len(cats),confusion_and_category_totals='PASS',same_record_scope='PASS',flip_balance='PASS',
    p3_manifest_identity='PASS',zero_support_categories=zero_support,
    source_supplement='2 frozen MLPs, 235424 source records each; original P2 inference_rows=131072; all saved P2 audit metrics reproduced exactly',
    new_training=False,new_platt_fit=False,threshold_search=False,stage_B_started=False,
    figures=sorted(p.name for p in (OUT/'figures').glob('*.png')))
(OUT/'final_validation.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')

scope_table=table(['Domain / Role','Record count','Benign','Attack','complete windows','Excluding tail'],[
    [s['domain']+'/'+s['partition'],f"{int(s['rows']):,}",f"{int(s['benign']):,}",f"{int(s['attack']):,}",s['complete_windows'],s['excluded_tail']] for s in scope])
asset_table=table(['Models','Dimensions','Platt coefficients','Intersection','Platt=0.5 corresponds to the Raw boundary'],[
    [r['model'],r['feature_count'],num(r['coefficient']),num(r['intercept']),num(r['raw_boundary_for_platt_half'])] for r in assets])
asset_doc=f'''# A: Assets and ratings quality

Project root directory: `{ROOT}`. This time, the output directory: `{OUT}`. This is an exploratory diagnosis of already unblinding data, not a new independent external verification.

## Read and Reused Materials

- `docs/execution_plan_v2.md`(v2.2-p7),`configs/execution_v2.json`,`configs/p7_v1.json`,`outputs/round1_v2/config_snapshot/freeze_record_revision_v2_1.json`.
- `outputs/source_v2/P2/P2_manifest.json`, `P2_validation.json`, six `models/*/training_manifest.json`, model entities, `calibrators/*.json`, `source_metrics/*.json`, and the original preprocessing statistics.
- `outputs/round1_v2/P3/P3_manifest.json`, `P3_validation.json`, `stage_report.md`, manifest predictions and cache fields define `src/nids/replay/predictions.py`.
- `outputs/outer_v2/P7/stage_report.md`, `data_audit.json`, `round1/metrics.json`, `identities.parquet`, and the original window manifest.
- `docs/Professor_Progress_Report_After_P7.md`, `docs/Professor_Progress_Report_After_P7_Followup.md`, `docs/p7_report_followup/outer_detector_classification.csv` and `scripts/report_p7_followup.py`.

The project has all the necessary materials. Alternative directory `/mnt/c/Users/ysy/Desktop/agent-nids/stage_report/` does not exist and is not blocked. Full features classification tables, f(u) directional audits, old CRE classification layers and the development of aluminum outer layer comparisons as a historical background replication have been developed; There was no complete analysis of the remaining three. This re-calculation of the Platt condition is used to pair with historical consistency verification.

## six freeze detector and probability identity

{asset_table}

The absolute path of the model, the path of the calibration apparatus and the loaded classes_ see [model_asset_audit.csv](model_asset_audit.csv). RF / XGBoost entities have classes_ equal to `[0,1]`; 0=Benign , 1=Attack , RF predict_proba 1 is classified as Attack. The binary:logistic booster output of XGBoost is P(Label=1). MLP is a single output BCEWithLogitsLoss, sigmoid corresponding to P(Label=1), with no classes_ or native classification predict interface.

P3 NPY is a structural array: `raw_probability` = - probability of an attack without Platt;`probability` = source domain fit and freeze Platt attack probability;`predicted_class`=`probability >= 0.5`;`u/h` is not involved in this probability quality assessment. Both probabilities are float64 storage; The basic model is based on the freeze float32 process.

Platt formula: First, calculate Raw to `[1e-7,1-1e-7]` and then calculate `sigmoid(coefficient × logit(Raw_clipped) + intercept)`. All six coefficients are positive; It is not possible to claim that the AUROC/AP must be strictly unchanged if the intersection converges with the tail score. The top-level equivalence boundary is an algebraic conversion of freeze to parameters, not a threshold search or threshold adjustment experiment.

Model view, features sequence, parameters, cache number, domain/model identifier consistent with P2/P3 manifest. The maximum Platt reconstitution error in the total evaluation record is 0 and the historical cached predicted_class mismatch is 0; The probability is limited and at [0,1]. There is no replacement of assets with a new training model.

## record quality with tail

{scope_table}

The full range of the source test is UNSW availability rank `[2130000,2365424)` with the end point 5,424; Testing and recording with the original source of P2. The goal is to use F1 → ToN , F3 → CIC , F2 → BoT for the T_heldout complete 10k window, which includes the initialization window, respectively. tail excludes 260 / 5,529 / 3,808 and 9,597 respectively. The records of the same domain in other development faults are not counted again in the total outer layer.

NPY in line r corresponds to availability rank r; Record identity as `domain:original_row` and the original line is from `outputs/round1_data_prep_v1/*_sorted_original_rows.npy`. This is a step-by-step verification of all P7 identities.parquet’s rank, original_row and record_id and verification of window spaces, endpoints and front drives through existing checked_windows. The six models share the exact same range as labels in each domain with Raw/Platt.

The original category reads directly `outputs/feature_store_v2/*_attack_code.npy` and attack_codebook of each manifest domain; The only category-by-category checks are binary labels, which corresponds to the category total, the Attack total and the TP/FP. Unmerged or redefined taxonomy. BoT has 1,615 bars in all Theft domains, but all are at the end of P7 excluded, with 0 counted and NA recalled. The other primitive categories are retained, with the Benign line judging the Attack ratio as FPR, without calling the attack category recalled.

## Fixed rules and historical correspondence

This time, Raw≥0.5 and Platt≥0.5 are both Attack, which is equal to 0.5 contained in Attack. The historical classification of P2 and P7 is also an obvious probability of ≥0.5; P7 uses the calculation of the corresponding conditions for Platt and 18 and all the comparative probability indicators are consistent within the 1e-12 gap. For more details see [historical_metric_comparison.csv](historical_metric_comparison.csv).

Native Interface Alternative: XGBoost sklearn.py using **Raw > 0.5** and Benign; The nvforest/_forest_inference.py of cuML takes argmax from the multi-column output (the ties takes the first class), and the single-column output uses a strictly greater threshold value, and there is no direct determination of the Attack threshold at this stage. This time, the original predict was not used to recalculate historical classification. Raw is equal to 0.5, see probability_validation.csv records: ToN RF-Traffic 675, RF-All49 445; CIC RF-Traffic 33 , RF-All49 45 , other MLP-Behavior 1; The rest is 0. Counts of Platt probabilities exactly equal to 0.5 are all 0. The original results were not mixed with the current or P7 indicator.

The source domain main table prefers to use P3 caching, keeping the same record quality as the P2. The P2/P3 probabilities of the two sets of MLP differ slightly, and the classification counts are exactly the same. To explain this difference, the freeze model and the original SourceModelData.predict_partition are additionally used to supplement the necessary reasoning to test the 235,424 line source for two sets of MLP, while maintaining the batch settings of P2 partition and **131,072**; There is no storage of new and incremental caches. All verified indicators of P2 are accurately reproduced, as evidenced by [source_p2_inference_audit.csv](source_p2_inference_audit.csv).

P2 pre-select the view and then preprocessing and sort the starting points by source partition; P3 Full 49 dimensional preprocessing after selecting the view and converting the contiguous into a whole area starting point. The two values/batch execution paths are different. MLP-Behavior Raw maximum lowest 2.3842e-7, Platt maximum lowest 2.6210e-7; MLP-All49 and 1.1921e-7 and 1.1254e-7 respectively; The classification reversal is 0. The P3 AUROC of Behavior is higher than the P2 0.000180388 and the AP 0.000994163; Microscopic scores can change the precise order of ties with nearby neighbors. The differences have been reconstructed, without regarding the specific layout and the respective contributions of the batch as separate mechanisms. The main table is still clear as the P3 recalculation result, and the P2 original value column does not rewrite history.

## indicator definition, calculation and resources

FPR=FP/(TN+FP),Recall=TP/(TP+FN),Accuracy=(TN+TP)/N,Balanced accuracy=(1−FPR+Recall)/2. AUROC and Average Precision in Attack=1 as positive classes; AP uses precision as a step-by-step threshold to add weight to recall and achieves precise ties grouping along the project, without using the gradient PR area.

ECE with 15 wide box: `[k/15,(k+1)/15)` , k=0 ... 13; Finally, the `[14/15,1]`, containing 0 and 1, enters the inner boundary into the right box. achieved at `min(floor(15p),14)`; ECE=  ⁇  Σ box |Σp−Σy| / N  ⁇ , equal to the sample weighted box error. Box counts, attacks and with probabilities and alternate ece_bins.csv. Brier=`mean((p−y)^2)`. Both are Attack probability masses, not CRE accuracy probability masses.

With a single true class, AUROC/balanced accuracy are NA according to available denominators; without Attack, AP/Attack recall are NA; empty-class recall is NA. All48 overall cases contain both classes, so overall metrics are defined. CSV FPR/Recall/Accuracy/BA values use0–1; their differences are percentage points, others retain original scales. Lower FPR/ECE/Brier is better; higher is better for the others.

The original prediction read from the mmap block on the 262,144 line, with two probabilities for only one set of models dealing with one domain at a time. The indicator directly replicates `src/nids/evaluation/round1_metrics.py:detection_metrics` to achieve CUDA, without replicating the full prediction caching or output of new flow trajectories. The probability rectangle graph only preserves the 50 box polynomials. Main computing equipment RTX A5000, time consuming {valid['elapsed_seconds']:.2f} Seconds (excluding audit supplementation, drawing and material verification)See validation.json for versions.

The initial launch stopped before metric computation because matplotlib was absent from the GPU environment; the error is preserved in run.log. The calculation is still in the agentnids environment; Drawings using `/home/karri/miniconda3/bin/python` (matplotlib 3.11.1) are already available, with no dependency or modification of the freeze environment. GPU Sandbox Restrictions are lifted by Executive Authorization. Running commands and logs see [commands.sh](commands.sh).

## total differences and historical documents

The total number of actual windows: 2,752+2,011+1,693=6,456; Record the total number of **27,520,000+20,110,000+16,930,000=64,560,000**. The old 65,560,000 is more than the actual 1,000,000 and is a total-plus-total error, not a new sample or a million-filtered record.

Affected narrative document: `docs/Project_Core_Conclusion_After_P7.md:7` still writes 65,560,000.`docs/Professor_Progress_Report_After_P7.md:117` and its English counterparts have been correctly interpreted;`docs/p7_report_followup/audit.json` is 64,560,000. P7 Folding data_audit/metrics is consistent with the actual window manifest without changing the folding indicator. These historical documents, freeze product or project configurations are not covered.

Phase A has been completed, no asset blocks. There are no new CRE, Strict-K, conformal prediction, P4–P7 overruns, coordinator modifications, P8 or B stages.
'''
(OUT/'asset_and_scope.md').write_text(asset_doc)

summaryrows=[]
for s in scope:
    d=s['domain']
    for m in models:
        r,p=lookup[d,m,'Raw'],lookup[d,m,'Platt']
        summaryrows.append([d,m,pct(r['FPR'])+'→'+pct(p['FPR']),pct(r['Recall'])+'→'+pct(p['Recall']),
            pct(r['Balanced_accuracy'])+'→'+pct(p['Balanced_accuracy']),num(r['AUROC'])+'→'+num(p['AUROC']),num(r['Brier'])+'→'+num(p['Brier'])])
summarytable=table(['domain','Models','FPR% Raw→Platt','Recall%','BA%','AUROC','Brier'],summaryrows)
focus=[]
for m in ['rf_Traffic','rf_All49','xgb_Protocol','xgb_All49']:
    for o in ['Raw','Platt']:
        r=lookup['ToN',m,o]
        focus.append([m,o]+[f"{int(r[k]):,}" for k in ['TN','FP','FN','TP']]+[pct(r['Accuracy']),num(r['ECE']),num(r['Average_precision'])])
focustable=table(['Models','The output','TN','FP','FN','TP','Accuracy%','ECE','AP'],focus)
deltas=load('feature_calibration_deltas.csv')
delta_rows=[]
for typ in ['All49_minus_split','Platt_minus_Raw']:
    selected=[r for r in deltas if r['domain']=='ToN' and r['effect']==typ and r['from_model'].startswith(('rf','xgb'))]
    groups=list(dict.fromkeys((r['from_model'],r['to_model'],r['condition']) for r in selected))
    for a,b,c in groups:
        rs={r['metric']:r for r in selected if (r['from_model'],r['to_model'],r['condition'])==(a,b,c)}
        delta_rows.append([('features '+a+'→'+b+' / '+c) if typ=='All49_minus_split' else ('calibration '+a)]+[f"{float(rs[k]['delta']):+.6f}" for k in ['FPR','Recall','Accuracy','Balanced_accuracy','AUROC','Average_precision','ECE','Brier']])
deltatable=table(['Difference conditions','ΔFPR pp','ΔRecall pp','ΔAcc pp','ΔBA pp','ΔAUROC','ΔAP','ΔECE','ΔBrier'],delta_rows)
fliptable=table(['ToN model','The real class','Raw incorrect→ Platt correct: % in category','Raw yes→ Platt wrong: % in category'],[
    [r['model'],r['true_class'],f"{int(r['raw_wrong_platt_right']):,} / {pct(r['raw_wrong_platt_right_rate'])}",f"{int(r['raw_right_platt_wrong']):,} / {pct(r['raw_right_platt_wrong_rate'])}"]
    for r in load('prediction_flip_summary.csv') if r['domain']=='ToN' and r['model'].startswith(('rf','xgb'))])

report=f'''# P7 A: Raw / Platt classification diagnosis

** has been completed: 4 field ×6 freeze detector ×2 output, total 48 conditions; It’s not like I don’t have any assets. The ** main table is completely recalculated for the current reading of the P3 probability cache without filling in the historical value. source domain makes the original batch of audit based on two other sets of MLP; Untrained, fit calibration device or adjustment threshold. This phase is an exploratory diagnosis of unblinding data and cannot be validated as a new independent outer layer. After completion, stop at A and wait for review.

## 1. scope, reuse and verification

{scope_table}

The target total **64,560,000 line/6,456 window **. 65,560,000 multiples of 1,000,000 of the old core abstract; The actual window matches the step-by-step indicator. The old document affected is `docs/Project_Core_Conclusion_After_P7.md:7` and the English version of the professor’s report has stated the correct value; This time, no old reports or freeze documents were covered. The source test retains the tail 5,424 clause, with the goal of excluding tail from the 9,597 clause; There are no duplicate records of the same domain in the different development folds.

The P7 table with a split view of ／All49 uses a **Platt Attack probability of ≥0.5**. This time, they replicated their identity control and matched the Raw, the original attack category, the probability quality, the rollover and the deviation values; No re-engineering of the old f(u), CRE substrate or complete analysis of the outer layer of silicon has been done. Asset, native predict rules, tail and numerical difference details see [asset_and_scope.md](asset_and_scope.md).

All confusion counts/category counts are consistent, the 12 conditions in each domain record exactly the same, the probability range passes with mapping; The total Platt reconfiguration error is 0; All comparison indicators for 18 historically corresponding to Platt conditions differ from P7 to 1e-12. The two sets of source P3/P2 MLP have small probability and sequence differences and are completely identical in classification; The original P2 set-up has accurately reproduced the nuclear indicator of P2, in addition to [source_p2_inference_audit.csv](source_p2_inference_audit.csv), without altering the historical results.

## 2. overall control: 6 detector results by domain

The values are **Raw → Platt**. FPR/Recall/BA as a percentage, AUROC/Brier as the original scale; The lower the Brier, the better. The total number of samples, the mixed matrix, Accuracy, AP, ECE, etc. are in [detector_raw_platt_metrics.csv](detector_raw_platt_metrics.csv). Recall is the rate of detection of binary attacks and the accuracy of classification of non-attack families.

{summarytable}

- **RF-Traffic17: ** source domain almost completely detected; ToN primitive FPR 42.712%, Recall 27.061% would have been a serious double-directional error, and Platt would have pushed FPR to 76.658%. The initial CIC error rate is also high, with Platt reducing BA from 57.538% to 49.360%. BoT Raw BA 85.107% , Platt although raising total Accuracy/Recall, pushes FPR from 21.455% to 84.024% , BA to 57.593% .
- **RF-All49: ** source domain is close to full; ToN in Raw has been significantly improved compared to Traffic in FPR/Recall, with Platt being only slightly classified but differing in ECE/Brier. CIC is still around 57% FPR, with Raw being misreported to be accompanied by a decrease in Recall; BoT Raw’s FPR, Recall and BA are both lower than Raw Traffic and cannot be called a whole better range.
- **XGBoost-Protocol17: ** source domain performs well; ToN , CIC , BoT , Recall , 0.052% , 0.588% , 0.794% and Platt respectively, were not reported. The corresponding Raw AUROC 0.755, 0.796, 0.812 indicates that certain sorting capabilities can coexist with near-inefficiencies under fixed thresholds.
- **XGBoost-All49: ** source domain is good; ToN and BoT Raw Recall/BA are significantly higher than Protocol; CIC is replaced by FPR with about 53.6%, Recall with about 69.1%, and AUROC is reduced from Protocol’s 0.796 to 0.535. CIC Platt improves ECE/Brier while reducing Accuracy, Recall and BA; The probability quality of ToN and BoT has not improved steadily.
- **MLP-Behavior15 : ** source domain Recall 69.101% → 70.940%, significantly lower than other high-performance source models; ToN about 11% about 13% about CIC about 5% about 6% about BoT about 6% about 9%ToN Platt improves ECE/Brier and BA; CIC deterioration of ECE/Brier and BA; BoT improves ECE/Brier and Accuracy, but FPR increases, BA decreases.
- **MLP-All49: ** source domain is almost complete; ToN Raw Recall 65.202%, BA 80.296%, significantly improved from Behavior, but slightly lower after calibration; CIC FPR is about 33.9%, Recall is about 11.5% and BA is less than Behavior; BoT Recall is still only about 13.8% and cannot be considered detector with all features.

## 3. Four conditions for RF–ToN and XGBoost–ToN

{focustable}

![RF ToN Attack Probability Distribution ](figures/RF_ToN_probability.png)

![RF ToN Confounding matrix ](figures/RF_ToN_confusion.png)

![XGBoost ToN Attack Probability Distribution ](figures/XGBoost_ToN_probability.png)

![XGBoost ToN Confounding matrix ](figures/XGBoost_ToN_confusion.png)

All probability charts use an equal width box **50 on [0,1], **, and integrate each real category internally into the total 1; The longitudinal range is consistent, with Benign 16,791,954 / Attack 10,728,046, and the hypotenuse 0.5. The final box contains 1; The probability graph is used to show the distribution, and the ECE is calculated according to a fixed 15 box. The confusion graph shows the ratio of the original count to the actual category.

** features effect =All49 − split view, with output; The calibration effect is =Platt − Raw, with the same model/view.** The first four items are percentage points (pp); the remaining items retain their original scales. ΔFPR/ΔECE/ΔBrier is negative for improvement and the rest is positive for improvement. These are observations under the freeze matrix model and the fixed 0.5 rule, which are not directly equivalent to features information loss or calibration failure of causal decomposition.

{deltatable}

See [feature_calibration_deltas.csv](feature_calibration_deltas.csv) for all four domains. Raw All49 BA gains are approximately39.52 pp for RF–ToN and32.71 pp for XGBoost, both before calibration.

{fliptable}

The sample subtractor is the complete record of the corresponding real class, not just the reversal record or the Raw error record. RF-Traffic’s Platt corrects 3,992,289 to Attack while making 5,700,222 to Benign from right to wrong; Therefore, the overall Accuracy decreases even if the Recall and BA rise. For all four real 48 classes, see [prediction_flip_summary.csv](prediction_flip_summary.csv).

## 4. Primary Attack Category: Unresolved Identification Gaps

Complete native category name, support number, Attack number and binary recall [attack_category_recall.csv](attack_category_recall.csv), sum {len(cats)} Lines, including Benign auxiliary lines and zero support categories. The following are the representative gaps:The effective identifier is descriptive and does not introduce a new threshold value.

- **UNSW : ** MLP-Behavior’s Generic (3,032 article) detects only 23.22% → 23.78% , Shellcode (266) 57.52% → 57.89%; Other models are closer to being fully detected. Small class Worms is only 21 and the proportion should not be exaggerated.
- **ToN: ** XGBoost-Protocol almost leaked to all nine classes. scanning (1,358,977) in three All49 models is only about 3.5% to 4.0%, although RF-Traffic can reach 68.90% → 98.38%, accompanied by overall high error rates; Therefore, all features gains are not covered by scanning. dos (203,456) is approximately 51.56%, 11.6% and 28.6% in three All49; xss (2,834,435) is about 51% to 55%, and there are still a lot of leaks. Raw RF-Traffic refers to ddos (4,141,256) only as 0.57% and Platt as 77.43%; Raw RF-All49 is now 91.91%.
- **CIC:** RF-Traffic detects 100% of FTP-BruteForce (386,720) and DoS_attacks-SlowHTTPTest (105,550), while all three All49 models, XGBoost-Protocol, and MLP-Behavior detect 0%. MLP-All49 for multiple DDoS classes and Bot for 0%; Infilteration (originally spelled 188,152) three All49 are only about 22.5% and one 33.9%. Per-category advantages cannot be summarized as all features being better.
- **BoT:** The three All49 models detect only about 7.64%, 6.24%, and 3.63% of Reconnaissance (1,695,132), respectively (Platt); RF-Traffic is equivalent to 38.56% → 95.05%, but misrepresentation is expensive. The DDoS of XGBoost-Protocol is 0% and DoS is about 1.24%; Both MLP feature views still miss many attacks. Theft originally had a 1,615 line, all falling into P7 excluding tail, the 0 rating at this stage, recalling NA, cannot be written as 0% detection or failure detection.

## 5. answers to six questions and explains the boundaries

** Confirmed Observation of **

1. **Were severe false alarms/misses already present in Raw? Yes.** RF–ToN Raw has high false positives and low recall; Protocol Raw recall is near zero in all three targets. These failures first occur before calibration.
2. Does **All49 improve before calibration? ToN does.** The Recall and BA of the three Raw architectures are improved; RF also significantly reduces FPR, while XGBoost/MLP yields a higher FPR. It is not possible to extend this observation to all domains or categories.
3. Is there a fixed classification that improves the probability quality of ** but deteriorates? Yes.** CIC XGBoost-All49 ECE 0.447286→0.443269,Brier 0.436376→0.428422; But Accuracy 49.341% → 49.321%, Recall 69.075% → 67.988%, BA 57.742% → 57.268%. BoT RF-Traffic and MLP-Behavior also showed improvement in ECE/Brier and deterioration in FPR/BA, but their Accuracy increased and indicators need to be described.
4. ** ranking is still available, but is the fixed threshold bad? Yes.** Three target domain XGBoost-Protocol’s Raw AUROC is about 0.755–0.812, while Recall’s <0.8%, BA is close to 50%. AUROC above random does not mean that calibration or a threshold value is available, nor does it mean that the optimal threshold value has been found at this stage.
5. **Which categories remain poorly detected?** See the previous section and full category table, particularly ToN scanning under all features, CIC FTP-BruteForce/SlowHTTPTest with 0 detections in multiple models, and BoT Reconnaissance under all features. The model has complementary gaps; It does not claim that there are non-empty categories that all models have in common that are completely missing.
6. Is **ToN improvement common?** CIC all-feature XGBoost has substantially worse FPR, AUROC, and Brier, while MLP has lower BA; BoT Raw RF-All49 is worse than Raw Traffic and XGBoost-All49 is clearly better. There are a lot of differences between the domains and the architecture.

** and possible explanation of ** consistent with observation

The - differs from the features view to the different coverage of the signal available in different regions, and is consistent with the ToN phenomenon where all features is improved in Raw. However, the comparison of two models that have been trained with freeze, respectively, cannot further ascertain whether a specific features or splitting scheme is the only reason.
- source Platt conversion does not match the target probability distribution, which may explain the difference in probability quality and classification cost between domains. RF-Traffic freeze intersection is positive, Platt 0.5 corresponds to Raw approximately 0.21607; Therefore, a large amount of Raw between that boundary and the record of 0.5 is converted to Attack, which is the determined decision mapping fact, not the objective optimal boundary.
- The whole Brier/ECE is weighted in proportion to the natural sample, and BA is weighted in proportion to equal weighting; It’s the opposite of change. The BoT data ratio explains the indicator weight difference, but ** cannot claim that it caused the failure of the ** model by a change in the class ratio alone.
- Platt input interrupts and limited accuracy may alter ties. CIC AUROC of MLP-All49 is about 0.206→0.298, BoT is about 0.491→0.563, and it cannot be written that the slope of Platt must remain AUROC. The original freeze mapping and tail disconnect counts have been fully verified, platt_clipped_tail_counts.csv and no new order changes have been made to fit.

** Problems unknown at this stage **

It is not possible to determine a single failure of causal, the contribution of each features independent, the specific mechanism by which the source calibration is mismatched, or the optimum FPR/Recall balance to be achieved by changing the threshold value. No target threshold searches, threshold adjustment experiments, category weighting experiments or new model selection. The indicator is an estimate of the descriptive points that have been recorded for unblinding, with no new confidence interval or significant conclusions. Theft cannot be estimated at the current complete windows as a whole; It’s not possible to push out the full flow of this category. The performance of new domains, follow-up periods, new CRE/coordination or start-up schemes is unknown.

## 6. Stage B decisions requiring review

Depending on the outcome of this stage, it is necessary to decide whether to continue the follow-up and focus on features/category coverage, probability calibration migration or fixed decision rules; Prioritized prior control of FPR, Recall, BA or probability quality to avoid bias between indicators. To continue the method comparison, it is necessary to first define which data is used only for exploration and which new independent data is used for confirmatory evaluation. It is also necessary to decide separately whether BoT Theft is to be included in the alternative range of tail. There is no substitute B method, threshold or start experiment at this stage.

Delivery indices: [ assets and calibration ](asset_and_scope.md), [ overall indicator ](detector_raw_platt_metrics.csv), [ original class recalling ](attack_category_recall.csv), [ deviation ](feature_calibration_deltas.csv), [ reversing ](prediction_flip_summary.csv), [ final verification ](final_validation.json), [ running order ](commands.sh). The actual script: run_stage_a.py, audit_source_and_clipping.py, plot_stage_a.py, build_report.py; Diary: run.log, source_audit.log, plot.log, report.log
'''
(OUT/'stage_A_report.md').write_text(report)
print(json.dumps({k:v for k,v in final.items() if k!='main_calculation'},ensure_ascii=False,indent=2))
