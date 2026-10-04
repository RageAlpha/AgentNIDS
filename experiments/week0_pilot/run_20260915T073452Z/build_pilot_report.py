from common import *
assert read(OUT/'validation.json')['status']=='PASS'
o=read(OUT/'overall_metrics.json');s=o['UNSW'];t=o['ToN'];q=read(OUT/'q_freeze.json');pf=read(OUT/'protocol_freeze.json');ev=read(OUT/'evaluation_complete.json')
ci={r['metric']:r for r in csvread(OUT/'bootstrap_intervals.csv')};time=csvread(OUT/'temporal_metrics.csv');cat=csvread(OUT/'attack_category_metrics.csv')
def pct(v,d=6):return 'NA' if v is None or v=='NA' else f'{100*float(v):.{d}f}%'
def num(v):return f'{int(v):,}'
def interval(k,d=6):
 r=ci[k];return f"[{pct(r['lower_95'],d)}, {pct(r['upper_95'],d)}]"
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows)
set_table=table(['Type of collection','UNSW source testing','ToN'],[[label,num(s[key]),num(t[key])] for label,key in [('{}','empty_count'),('{Benign}','benign_singleton_count'),('{Attack}','attack_singleton_count'),('{Benign,Attack}','multi_label_count')]])
rate_table=table(['Indicators','UNSW source test (descriptive)','ToN','ToN 95% block spaces'],[[label,pct(s[k]),pct(t[k]),interval(k)] for label,k in [('prediction_set_coverage','prediction_set_coverage'),('Benign prediction_set_coverage','benign_prediction_set_coverage'),('Attack prediction_set_coverage','attack_prediction_set_coverage'),('acceptance_rate = singleton_rate','acceptance_rate'),('multi_label_rate','multi_label_rate'),('empty_set_rate = rejection_rate','rejection_rate')]])
rej_table=table(['Indicators','Source testing','ToN','ToN 95% block spaces'],[[label,pct(s[k]),pct(t[k]),interval(k)] for label,k in [('P(raw_error | R)','raw_error_given_rejected'),('P(raw_error | not R)','raw_error_given_accepted'),('P(R | raw_error)','rejected_given_raw_error'),('P(R | raw_correct)','rejected_given_raw_correct')]])
sel_table=table(['Indicators','Source testing','ToN'],[[label,pct(s[k]),pct(t[k])] for label,k in [('selective_risk','selective_risk'),('Real Benign acceptance rate','benign_acceptance_rate'),('Real Attack acceptance rate','attack_acceptance_rate'),('Benign FPR has been accepted','accepted_benign_FPR'),('Attack false negative rate has been accepted','accepted_attack_miss_rate'),('It’s the right Attack / all Attack.','effective_attack_detection'),('Attack / all Attack is incorrectly accepted as Benign.','wrong_accepted_attack_fraction'),('Attack abstention rate','attack_abstention_rate'),('Full acceptance of Raw error rate','raw_error_rate'),('Raw Attack withdrawn','raw_attack_recall')]])
time_table=table(['Paragraph / between the original rank','Benign/Attack','Accept the (TN/FN)','Acceptance rate','prediction_set_coverage','selective risk','effective Attack detected'],[[str(r['segment'])+' / ['+r['rank_start']+','+r['rank_end_exclusive']+')',num(r['benign'])+'/'+num(r['attack']),num(r['accepted'])+' ('+r['TN']+'/'+r['FN']+')',pct(r['acceptance_rate']),pct(r['prediction_set_coverage']),pct(r['selective_risk']),pct(r['effective_attack_detection'])] for r in time])
time_rej=table(['Section 1','P(error|R)','P(error|not R)','P(R|error)','P(R|correct)','Benign acceptance rate','Attack acceptance rate'],[[r['segment'],pct(r['raw_error_given_rejected']),pct(r['raw_error_given_accepted']),pct(r['rejected_given_raw_error']),pct(r['rejected_given_raw_correct']),pct(r['benign_acceptance_rate']),pct(r['attack_acceptance_rate'])] for r in time])
cat_table=table(['ToN raw category','All supported.','Acceptance','Accepting Attack correctly','Put it wrong as Benign.','abstention','Accept the risk'],[[r['category'],num(r['rows']),num(r['accepted']),r['TP'],r['FN'],num(r['rejected']),pct(r['selective_risk'])] for r in cat if r['domain']=='ToN'])
scan=next(r for r in cat if r['domain']=='ToN' and r['category']=='scanning')
gap=t['rejection_rate']-s['rejection_rate']
gate=f'''# Week 0 gate assessment

** Mechanism: NO_GO_EVIDENCETechnical verification: PASSThis judgement is based on a joint, artificial audit of pre-registered evidence, not a new value threshold.**

## 1. set changes appear, but not enough to prove useful
ToN rejection rate from source test {pct(s['rejection_rate'])} Up and down.{pct(t['rejection_rate'])} Descriptive defects {gap*100:.6f} One percentage point; It’s all about the change from the empty collection. q={q['q_decimal']}<0.5, more than labels, is a structural zero, so the algorithm has not failed.
Rejected {num(t['rejected'])} It’s only accepted at the same time {num(t['accepted'])}/{num(t['rows'])} The article states that almost all business records are excluded. The vacuum cluster cannot be defined as Unknown Attack: the original Benign is also almost completely rejected.

## 2. Distinction error directions do not support the roadmap requirements
Refusing the Raw error rate {pct(t['raw_error_given_rejected'])} Accept the Raw error rate {pct(t['raw_error_given_accepted'])}; Variable value {t['raw_error_rate_gap']*100:.6f} One percentage point, paired with block bootstrap between block [{float(ci['raw_error_rate_gap']['lower_95'])*100:.6f}, {float(ci['raw_error_rate_gap']['upper_95'])*100:.6f}] percentage points.
P(R|error)={pct(t['rejected_given_raw_error'])}，P(R|correct)={pct(t['rejected_given_raw_correct'])}; Variable value {t['rejection_rate_error_correct_gap']*100:.6f} One percentage point, [ range {float(ci['rejection_rate_error_correct_gap']['lower_95'])*100:.6f}, {float(ci['rejection_rate_error_correct_gap']['upper_95'])*100:.6f}] percentage points.
Refusal to include {num(t['rejected_raw_errors'])} A Raw error, but at the same time loss {num(t['rejected_raw_correct'])} The Raw is accurate. The high total number of catches is mainly accompanied by almost total rejection and cannot be called targeted catching errors.

## 3. detection performance does not support continuation
Accept the TN/FP/FN/TP= set {t['TN']}/{t['FP']}/{t['FN']}/{t['TP']}; Accept the risk {pct(t['selective_risk'])} The 95% range {interval('selective_risk')}.
effective attack detected as 0; The actual Attack that is accepted for 616 is all mistaken for Benign and the remaining Attack is abstention. Benign FPR=0 has been accepted based only on 115 clause Benign, which cannot be overshadowed by the cost of rejecting almost all positive records.
The source test accepts risk as 0, but only detects {pct(s['effective_attack_detection'])} All of Attack attacked abstention.{pct(s['attack_abstention_rate'])}. Therefore, low source domain and low risk must also be combined with attack to explain.

## 4. scanning and time performance
scanning support {num(scan['rows'])}(i) accepting only the 3 and misrepresenting it entirely; The 0 is correct. Four sections accept 20/265/1/445, risk 0%/80.377358%/100%/90.337079%; All phases correctly accept Attack as 0.
The first zero risk can’t be explained as an early success with only 20 being the correct Benign and not accepting Attack. Class and time results are descriptive, supporting the failure of the function of this condition, but not solely demonstrating that time drift on causal or future would be the case for all time periods.

## 5. state boundary
- does not select IMPLEMENTATION_INVALID: model / enters identity, q definition, numerical contract, total quantity and phase A, independent polymer /2000 replication is passed.
- opts for NO_GO_EVIDENCE: the set is changed, but almost completely rejected, more zone errors are accepted, the effective attack is detected as 0, and the combined evidence is insufficient to support this path to continue.
- did not select POTENTIALLY_PROMISING_REVIEW_REQUIRED: The required error distinction with the available test performance did not occur.
- does not substitute INCONCLUSIVE for the overall conclusion: some categories of / have a time fraction of less than or equal to zero and retain NA; These constraints do not completely erase the fact that the calculation of TP=0 is almost entirely rejected. bootstrap does not constitute future risk Guarantee.

Aluminum meaningfully more Aluminum errors concentrated Aluminum has no original value threshold; No subsequent pre-registration thresholds. 99%+singleton is just an example of roadmap, this time another no-go: almost entirely empty set. The range not exceeding 0 itself is not a real value standard.
epsilon=.10 is not equal to 10% selective risk, does not inherit the C2 5% risk /10% acceptance threshold, does not adjust q to meet the source test or ToN.

** recommends reporting the negative pilot results to the professor in roadmap and then determining the direction; It’s not pilot confirmation and it’s not Week1. This is the first time anyone has sent a message.**
This result is for freeze RF-All49/ToN/VariantA/Strict-0/epsilon=.10 only; It does not prove that other models, Mondrian, or other epsilon coordinates will fail, nor does it constitute authorization for these experiments.
'''
(OUT/'gate_assessment.md').write_text(gate)
report=f'''# Week 0 Feasibility Pilot phase report

** Conclusion: NO_GO_EVIDENCE; Technical implementation and verification of PASS. The prediction set for the ** source calibration is almost entirely empty after the ToN migration; Accept only the 731 clause, where 616 is an incorrect attack, and correctly accept Attack as 0. No support for continuing to enter Week1 under current RF-All49/ToN/VariantA Strict-0 conditions.

Catalogue: `{OUT}`. The official protocol is week0-pilot-v1.0; Only one freeze detector, one goal, epsilon=1/10, no weight training, no extra softmax, no Platt/f(u)/C1/C2 threshold and goal 400 calibration.
This wheel is the exploratory pilot after the P7 and A cylinders C2 unblinding, not the first sighting of the target or the new independent external verification. The Windows summary is not currently synchronized; See provenance/background_summary.json, using the previous stages of reporting as a background, not participating in computing, not blocking.

## 1. protocol, input and execution order

- protocol freeze:{pf['frozen_at_utc']}.
- q calculation starts with:{q['computation_started_at_utc']}; q freeze:{q['frozen_at_utc']}.
- Source Testing /ToN Assessment begins:{ev['started_at_utc']} In the past, there has been no access to assessing probability for collective computation.
- protocol_freeze.json and q_freeze.json respectively save the / protocol code / input and the q hash; The evaluation process alone is loaded with freeze q only, without calling the threshold option. q, epsilon, range and algorithm are not modified according to results.
- All preflight source contents are hashed together; The original model is consistent with the environmental version, and the successful replication of the original RF loading record with preflight. CPU NumPy2.4.6/float64 performs sets and aggregates without running new model reasoning.
- source calibration, source test, ToN limited verification of each reading block, [0,1] and float32 lossless improvement of the contract; Check the / interrupted input. The original 49 features / preprocessing retains the exclusive source of S_train.
- source identity complete ranking and partition isolation review; ToN identity / window file hash is consistent with preflight full quantity identity verification. The window was not rebuilt. The current code / log does not find a S_prob_cal driven Raw RF selection pattern, and does not claim to exclude unrecorded human attempts.

| partition | rank Semi-open range | record number | Benign | Attack | tail |
|---|---|---:|---:|---:|---|
| Source S_prob_cal | [1420000,1780000) | 360000 | 325600 | 34400 | all using |
| Source S_source_test | [2130000,2365424) | 235424 | 215407 | 20017 | Retain 5424 records |
| ToN P7 T_heldout | [0,27520000) | 27520000 | 16791954 | 10728046 | excluding original 260 clause |

ToN contains the previous 100000 line and the initialization window, 2752 is the original 10k window, which is the same as the A phase, without comparing C1 shrinkage scope calculations.

## 2. Source calibration q and ties

q decimal point **{q['q_decimal']}**；float64 hex **`{q['q_serialized']['value']}`**. n=360000 , k=ceil((n+1)×9/10)=324001 (one group), and zero group 324000. Directly upgrades k items without using linear inputs.

| calibration fractional relationship | items number |
|---|---:|
| a<q | {q['scores_less_than_q']} |
| a=q | {q['scores_equal_q']} |
| a>q | {q['scores_greater_than_q']} |

The lowest calibration {q['score_min']} The biggest.{q['score_max']} mean {q['score_mean']:.12g} In total,{q['score_unique_count']} A different value. After containing the whole group of q, calibration is the real labels into which 326310/360000=90.641667% is inserted. 38,470 points q ties are all Benign; There are no random disassembly groups or test sets determined by labels.
The two types of source calibration are integrated, respectively, 313934/325600=96.417076% and 12376/34400=35.976744%; This means that overall about 90% is not equal to Attack about 90%. The difference between the two types of source testing continues this fact.
The source test covers the overall 91.309297% and is not mechanically error-prone; independent score hash / location / ties number and save recovery verified. It is observed that lower Attack coverage is an important limitation of standard marginal VariantA and is not correlated with q. This wheel does not offer time partition interchangeable assurance.

## How the 3. source and target prediction sets change

{set_table}

{rate_table}

mean_set_size: Source {s['mean_set_size']:.12g}、ToN{t['mean_set_size']:.12g} It’s equal to singleton_rate+2×multi_label_rate. The source acceptance_rate is the same as the prediction_set_coverage because all the predictions are correct; The two quantities differ substantially on ToN and must not both be abbreviated as coverage.
ToN - Source rejection rate less than **{gap*100:.6f} A ** percentage point, descriptive only, and does not give a source-specific target matching range.
q<0.5 makes the binomial set structural as 0; ToN without {{Attack}} Single elements, the vast majority of both categories of candidates do not meet the threshold. The vacuum is the wheel’s refusal to move, not directly named Unknown Attack.

## 4. denies whether effective captures the error

{rej_table}

ToN refuses to group Raw error {num(t['rejected_raw_errors'])}/{num(t['rejected'])} Accepting the Raw error set {num(t['accepted_raw_errors'])}/{num(t['accepted'])}。
P(error|R)−P(error|not R) = **{t['raw_error_rate_gap']*100:.6f} A percentage point ** , 95% range [{float(ci['raw_error_rate_gap']['lower_95'])*100:.6f}, {float(ci['raw_error_rate_gap']['upper_95'])*100:.6f}] percentage points.
P(R|error)−P(R|correct) = **{t['rejection_rate_error_correct_gap']*100:.6f} A percentage point ** , 95% range [{float(ci['rejection_rate_error_correct_gap']['lower_95'])*100:.6f}, {float(ci['rejection_rate_error_correct_gap']['upper_95'])*100:.6f}] percentage points.
Both values are calculated directly in each shared copy. the direction of misinterpretation does not support the error concentration in the rejection zone; In the meantime, I’m going to try to find a way to make the most out of it, even though I’ve denied almost all of it.{num(t['rejected_raw_correct'])}/{num(t['raw_correct'])} The Raw is accurate. Capturing many errors alone does not establish effectiveness.
The record number for the four sets respectively is /Raw correct / error and real divisor see set_type_error_counts.csv; Raw baseline confusion counts and phases A are all exactly the same, see historical_raw_comparison.csv for details.

## 5. Low risk, misreporting and attack leaks

The source accepts TN/FP/FN/TP=207907/0/0/7057; ToN accepts =115/0/616/0.

{sel_table}

ToN generally does not have any useful low risk reception zone: risk 84.268126% , 95% range {interval('selective_risk')}. Attack **0/10728046** is correct, 616/10728046 is wrong, abstention 10727430/10728046 is wrong. 7,606,499 and Attack can be detected by all Raw receptors; These correct attack predictions were not accepted after the q was fixed.
The source test risk 0 cannot be called a successful test alone: only 7057/20017 with Attack and abstention with 12960 with Attack. ToN has received only 115 as a fraction of Benign FPR=0; The remaining 16,791,839 Benign is rejected. Zero FP is not a practical error reporting control proof on a complete flow.
All proportions and numerator/denominator exist separately in the CSV indicator. The denominator of the zero support or zero acceptance rate is 0 time marked NA; The number is still 0. ToN effective Attack detects [0,0] in the bootstrap range from the current TP blocks all as 0, which cannot be explained as the uncertainty law of the future as a whole.

## 6. Primary attack category, focusing on scanning

{cat_table}

scanning total 1,358,977 clause (full P7 range), only 3 clause accepted and all misplaced as Benign, correct detection of 0 , abstention 1,358,974 clause. Its Raw background is called {pct(scan['raw_attack_recall'])} However, this condition does not preserve any correct detection.
password/ransomware/Backdoor zero acceptance and selective risk for NA; I can’t write 0 risk. The taxonomy remains as it was; The Benign line is not a recall attack class. Both classes and primitive classes are numbered / covering /, rejecting /, and accepting the fractions class_metrics.csv and attack_category_metrics.csv.

## 7. time stability (descriptive)

Each of the four 688 windows always uses the same q.

{time_table}

{time_rej}

All four sections do not correctly accept Attack. The zero risk in the first paragraph comes only from 20 clause Benign, Attack accepts 0 and Attack false negative rate accepts NA. The third paragraph accepts only 1 error Attack and does not accept Benign and has accepted Benign FPR as NA. The second / is four risk 80.377358%/90.337079%, respectively.
This shows very little sampling and risk is unstable and overall functional failure throughout the stages; The first paragraph is not considered a successful starting point, nor is the causal drift asserted separately based on the difference between the paragraphs. For the full / category, see temporal_metrics.csv and temporal_attack_category_metrics.csv and no new CI category.

## 8. statistical quality

ToN by the adjacent 10 window block, 276 block, and 2 window. Each copy has extracted 276 bits, 2000 times; PCG64 seeds 20260915, index, multiplicity, block range and hash have been preserved. All indicators use the same matrix to recalculate the fraction and ratio after the aggregate integer counts, without repeating the C1 matrix.
95% percentile uses the linear quotient and is only used for bootstrap estimates, not for q conservation. All reporting overall indicators have 2000 replicas of effective, and the zero denominator rule is still implemented according to NA and synthetically checked. The difference value interval is calculated directly by the same duplicate.
Source test 23 complete windows +5424 tail by original block length only 3 blocks, does not satisfy at least 20 blocks, so do not source CI, do not shorten block length, do not remove tail. Time and category are descriptive. ToN and the near-source rejection rate claim to match CI.
The interval conditioned on the freeze model, the fixed source q and the current observed sequence; Does not contain the uncertainty of retrieving the source calibration set, does not constitute a future domain coverage or a risk guarantee.

## 9. Technical testing and gate

validation.json for PASS: necessary input hashes and all old files size/mtime remain unchanged, q is from S_prob_cal only and freeze is early in evaluation; Four sets of / proportions / size constant, single elements consistent with Raw, error constant, category / time / window aggregation, phase A and range Raw confusion counts are passed.
All 2000 replicas use direct indexed whole blocks and independent replicas and verify shared multiplicity, total ratio subtractors, effective replicas and intervals; The independent metric concentration covers the whole, per window, per class and time period.
Technical PASS is not equal to mechanism GO. Pre-registered combined evidence supports **NO_GO_EVIDENCE**: Although the target vacuum cluster increases, almost all rejections, accepting more zone errors, effective attacks detect 0. No additional minimum impact thresholds; The 0 range does not pass the standard separately, nor does C2’s 5% risk /10% acceptance rate.
Detailed four states exclusion and judgement see gate_assessment.md. Failure by this pilot to deduce the / objectives, Mondrian, other epsilon or detector coordinates from other models will inevitably fail; It is also not possible to confirm an unknown attack by the vacuum cluster or to guarantee that future covers / risk.

## 10. Delivery and Stop

The main files are: protocol.json/protocol_freeze.json, source_hashes.csv, q_freeze.json, calibration_summary.json, prediction_set_metrics.csv, set_type_error_counts.csv, rejection_diagnostics.csv, selective_metrics.csv, attack_category_metrics.csv, temporal_metrics.csv, bootstrap_intervals.csv, sharing the bootstrap matrix, necessary window assembly, gate_assessment.md, validation.json, all scripts / commands / diary.
The number of window counts preserves only the primary window × original category × true labels ×Raw predicts the × collection type of mercury polymer, source tail single column without pretending to be complete windows; No complete probability caching or outflow trajectory has been copied.
** official Week0 stop after completion. According to roadmap, the results of this pilot should be reported to the professor and obtained confirmation before deciding whether to enter Week1. This time, no Week1 or other model / targets, Mondrian or epsilon have been scanned, coordinated or assembled, and no messages have been sent.**
'''
(OUT/'stage_Week0_pilot_report.md').write_text(report)
write('gate_assessment.json',dict(status='NO_GO_EVIDENCE',implementation_valid=True,decision_basis='Joint evidence review; no new numerical thresholds',source_target_rejection_gap_descriptive=gap,target_raw_error_rate_gap=t['raw_error_rate_gap'],target_rejection_error_correct_gap=t['rejection_rate_error_correct_gap'],accepted=t['accepted'],TP=t['TP'],FN=t['FN'],professor_confirmation_received=False,Week1_authorized=False))
(OUT/'metric_dictionary.md').write_text('''# Metric and aggregate contract

All rates are proportions (0–1), except mean_set_size (0–2). Reports display percentages explicitly. Each ratio has <metric>_numerator and <metric>_denominator; missing denominators give CSV NA/JSON null. Counts remain zero when no records exist. Gaps are differences of the named component ratios and are NA if either component is undefined.

Prediction-set coverage = true label contained in set. Acceptance = exactly one label. Raw classification = p>=0.5. Accepted prediction = unique set label, checked equal to Raw. Set code 0={}, 1={Benign}, 2={Attack}, 3={Benign,Attack}. R=set code 0 or3. True labels and taxonomy are accessed after set generation, used only for aggregation.

Each NPZ window_counts tensor axes are window/category/true_label/raw_pred/set_code. For UNSW last slice is retained 5424-row partition tail, not a new full window. For ToN all 2752 slices refer to original IDs in evaluation_window_references.csv. Schema and original codebooks in aggregation_schema.json. Counts allow exact recomputation of all requested metrics and block replicates; no flow-level trace stored.

selective_risk=(FP+FN)/accepted; accepted_benign_FPR=FP/(TN+FP); accepted_attack_miss_rate=FN/(TP+FN); effective_attack_detection=TP/all_true_Attack; wrong_accepted_attack_fraction=FN/all_true_Attack; attack_abstention_rate=(all_true_Attack-TP-FN)/all_true_Attack. For a Benign-only stratum the three Attack rates are NA, not zero.

raw_error_rate_gap=P(error|R)-P(error|not R); rejection_rate_error_correct_gap=P(R|error)-P(R|correct). Both are recomputed inside the same bootstrap replicate. Source-target rejection gap is descriptive only.
''')
val=read(OUT/'validation.json');val.update(mechanism_gate='NO_GO_EVIDENCE',calibration_fixed_q_audit=read(OUT/'calibration_verification.json')['status'],stage_report=str(OUT/'stage_Week0_pilot_report.md'),gate_report=str(OUT/'gate_assessment.md'))
write('validation.json',val)
print('Reports complete: technical PASS, mechanism NO_GO_EVIDENCE; stopped before Week1')
