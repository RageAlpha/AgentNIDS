from pathlib import Path
import csv,json,hashlib,ast
from datetime import datetime,timezone
import numpy as np
O=Path(__file__).resolve().parent;R=Path('/home/karri/projects/NIDSProject_Professor_Run');A=R/'outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z';L=R/'outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z';P=R/'outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z'
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def table(n,x):
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
# Source protection and provenance: actual current bytes vs preceding independent audit.
known={r['path']:r['sha256'] for r in rows(L/'source_hashes.csv')}
paths=[R/'datasets/NF-UNSW-NB15-v3.csv',R/'datasets/NF-ToN-IoT-v3.csv',R/'outputs/source_v2/models/rf_All49/model.pkl',R/'outputs/source_v2/models/rf_All49/training_manifest.json',R/'outputs/source_v2/preprocessing/preprocessing_stats.npz',R/'outputs/feature_store_v2/feature_store_manifest.json',P/'protocol.json',P/'q_freeze.json',P/'gate_assessment.md',R/'src/nids/models/base_models.py',R/'src/nids/data/preprocessing.py',R/'src/nids/data/feature_store.py']
for d in ['UNSW','ToN']:
 paths +=[R/f'outputs/feature_store_v2/{d}_X.npy',R/f'outputs/feature_store_v2/{d}_label.npy',R/f'outputs/prediction_cache_v2/{d}/rf_All49.npy',R/f'outputs/round1_data_prep_v1/{d}_sorted_original_rows.npy',A/f'{d}_sample_features.npz']
paths += [A/'sample_freeze.json',A/'inference_validation.json',A/'probability_score_diagnostics.csv']
hashrows=[]
for p in paths:
 h=sha(p);old=known.get(str(p));hashrows.append(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=h,prior_hash=old or 'NOT_REGISTERED',matches_prior=h==old if old else 'NO_PRIOR_HASH'))
table('source_hashes.csv',hashrows)
assert all(r['matches_prior'] is True for r in hashrows if r['prior_hash']!='NOT_REGISTERED')
raw=rows(O/'full_raw_four_feature_verification.csv');assert all(int(r['mismatches'])==0 for r in raw)
reuse=rows(O/'source_exact_feature_reuse.csv');assert all(int(r['candidate_not_exact'])==0 for r in reuse)
q=json.loads((P/'q_freeze.json').read_text());cal=rows(O/'calibration_rank_decomposition.csv');assert sum(int(r['score_below_q']) for r in cal)==q['scores_less_than_q'] and sum(int(r['score_equal_q']) for r in cal)==q['scores_equal_q']
fi=rows(O/'frozen_model_feature_importance.csv');ttl=sum(float(r['importance']) for r in fi if r['feature'] in ['MIN_TTL','MAX_TTL']);packet=sum(float(r['importance']) for r in fi if r['feature'] in ['MIN_IP_PKT_LEN','SHORTEST_FLOW_PKT'])
validation={'status':'COMPLETED_WITH_SCOPED_CAUSAL_LIMITS','completed_at':datetime.now(timezone.utc).isoformat(),'q_reselected':False,'model_retrained_or_modified':False,'new_pilot_or_Week1':False,'new_model_predict_proba_calls':0,'new_read_only_frozen_tree_traversals':53813,'full_four_feature_elements_verified':4*(2365424+27520000),'full_four_feature_mismatches':0,'source_feature_overlap_exact_elementwise_candidate_checks':True,'source_calibration_train_feature_matches':45252,'source_test_train_feature_matches':28401,'train_cal_test_complete_raw_record_cross_matches':0,'source_entities_shared':'all source/destination IPs in calibration/test seen in corresponding train role','TTL_training_impurity_importance_sum':ttl,'two_packet_length_importance_sum':packet,'prior_input_hash_mismatches':0,'historical_operations':'NOT_VERIFIED; no new documentary evidence','all49_raw_reconstruction_scope':'prior 53813 samples; additional four columns full domains only','full_backend_reinference':'NOT_PERFORMED; previous sampled backend equality remains evidence','numeric_cause':'exact mixed-score rank + discrete near-pure source leaves; target leaf disagreement under shifted feature distributions','causal_limit':'TTL mechanism supported by full distributions, model structure and sampled paths; no intervention attribution or full acquisition-pipeline explanation','NO_GO_EVIDENCE_supported_current_configuration':True}
dump('validation.json',validation)
report='''# Week Location of 0 probabilities and upstream causes of anomalous devaluations

## Conclusions

The analysis advances from possible domain shift to a localized mechanism chain: **source labels correlate strongly with environmental features such as TTL/packet length, and the frozen RF uses these features extensively, producing near-pure-class probabilities; The positive mixed calibration score is taken to very low q on the large ties group; ToN These features distributions change, and the output of the leaves of the tree is no longer nearly consistent, and therefore cannot meet the extreme singleton threshold.**

The first two parts (score ranks,leaf averages) have direct numeric evidence. TTL is an important clue,but TTL alone causing all transfer failure is not causally identified. Cross-source-partition duplicate feature vectors/shared entities are confirmed,not merely possible. No project sorting,imputation or probability-cache-write error was found. Missing original collection/export processes and unrecorded human actions remain unverified.

This round does not modify q, epsilon, model, evaluation range or gate, no training, modification, features, removal, dissolution, re-implementation of pilot or Week1. All results are read-only. Tree structure derivatives are used only to explain the original model and do not produce pilot results as an alternative model.

## 1. Why is q so small: Directly positioned

360000 in calibration, Benign325600 in 90.4444% and Attack34400 in 9.5556%. Fixed a group in order k=324001, without re-selecting q.

| labels | percentage <q | percentage =q | percentage >q | total |
|---|---:|---:|---:|---:|
| Benign | 275464 | 38470 | 11666 | 325600 |
| Attack | 12376 | 0 | 22024 | 34400 |
| combined | 287840 | 38470 | 33690 | 360000 |

The 324001 number is in the larger ties group of 38,470 class Benign, so q=`3.145372886592668e-07` is. This is the direct result of a mixed sequence of all source calibration real scores, not a scoring value, Platt or softmax or an incorrect index.

Specific records: UNSW rank1420799 / original_row405166 , real Benign , MIN/MAX_TTL is 31 . In the 300 tree, the 294 tree falls on pure Benign leaves, leaving the 6 tree with a probability of attack of about `2.685389e-6, 1.434354e-6, 1.665759e-6, 3.699967e-6, 8.403793e-5, 8.377890e-7`. mean is about `3.14537290933e-7`, the original float32 RF output is freeze q. This recording path for the 295 tree uses TTL.

This explains the model source of small probabilities: almost all trees are considered Benign and a few large leaves contain very little Attack mass. It’s not a normal probability error to be reduced to 10^-7. The original float32 request and the reference float64 mean will differ in the endpoint, with reference to the mean only explaining the mechanism, without replacing the original probability or threshold.

Mixed calibration is not a class calibration: overall 90.6417% coverage allows only Attack to be covered by 35.9767%. The numerical advantage of Benign determines the ranking position together with the extreme score; It is not possible to simplify addiction as long as the category is not balanced, it is necessary to reduce the q.

## 2. source domain Why it’s so easy to generate extreme probabilities: specific features clues

Read directly the importance of the freeze model and the 300 tree instead of reinterpreting the fit model:

| features | Training of impurities importance |
|---|---:|
| MIN_TTL | 25.9759% |
| MAX_TTL | 25.6657% |
| MIN_IP_PKT_LEN | 13.6217% |
| SHORTEST_FLOW_PKT | 11.7837% |

TTL is two columns plus 51.6416% and four columns plus 77.0471%. The importance of the features and the training distribution is influenced, and it is not possible to read addiction to explain the target error of the 77%. However, it directly demonstrates extensive use of these features in the model structure.

The results of the full-scale distribution verification:

| domain / category | MIN_TTL main value of |
|---|---|
| source training Benign | 31 : 99.0861% |
| source training Attack | 254 : 75.2937%; 62:24.5278% |
| Source calibration Benign | 31 : 97.5120% |
| source calibration Attack | 254 : 75.6279%; 62:24.2413% |
| ToN Benign | 0:91.5883%; 64:5.6178% |
| ToN Attack | 64:68.1293%; 0:29.0155% |

The length has also changed: source training Benign has MIN_IP_PKT_LEN=52 as 80.5513% and source training Attack as 40 (66.1028%) or 0 (27.3189%); ToN Attack is 52 instead of 53.5827% and ToN Benign is 0 (61.8257%).

**Evidence-supported interpretation:** source labels strongly associate with traffic-generation/network-environment features;the model uses patterns persisting in source calibration/test but absent from same-label ToN distributions. This supports environment-pattern dependence/shortcut risk,without proving illegal TTL label leakage or why collection systems produced these values.

## 3. ToN Why can’t the Attack threshold be reached: Leaf output positioned

A fixed `1-p<=q` determination requires p to be approximately `0.9999996854627113`. The maximum p=`0.989444375038147` is observed at ToN.

In the verified sample, there is a record of the maximum total value: ToN rank8842664 / original_row8841709, real Attack. Its 300 tree output consists of: 296 pure Attack, 2 pure Benign, and 2 pure 1/3 and 1/2. mean is approximately:

`(296 + 0 + 0 + 1/3 + 1/2) / 300 = 0.989444444…`

The original float32 RF output is consistent with the cache. Even though it is considered Attack by the vast majority of trees, it is still not enough to pass the threshold of approaching 1. This does not require assuming that the caching is clip to 0.98944; The original model structure has explained the maximum value record.

53,813 also supports the same mechanism for tree path traversal in both fixed audit samples: median for the number of pure Attack trees in source Attack is approximately 297; ToN Attack is about 185, while the number of pure Benign trees median is about 98. It is a description of the sample mechanism, not a statistical or independent performance evaluation.

Source also contains Raw-correct/rejected examples:source Attack rank1421299 has p=`0.9999995827674866`,297 pure-Attack trees/3 mixed trees. It exceeds0.5 but remains below the singleton threshold. Rejecting source attacks does not mean Raw cannot recognize them.

Referring to the float64 tree, the maximum output of mean is approximately 1.37e-7 with respect to the original float32, only for the diagnosis of different pathways of aggregation; ToN The highest candidate Attack score is 0.0105556, which is far from the same rating. The conclusion that the original GPU was reconstructed and consistent with the phase-by-phase caching was unchanged. It is not possible to call the endpoint difference of reference mean a caching error, nor to change the boundary algorithm accordingly.

Is ## 4. TTL=0 / abnormality caused by our preprocessing?

** is not the result of writing or filling in the four columns that have been checked.** has been expanded to the original CSV full-size four-column closure: UNSW2365424 line, ToN27520000 evaluation line, original_row mapping element-by-element checking for MIN_TTL, MAX_TTL, MIN_IP_PKT_LEN, SHORTEST_FLOW_PKT; There is a 119541696 element with 0 difference.

The 15,379,466 clause of ToN is Benign (91.5883%), MIN_TTL=0 (3,112,795), Attack (29.0155%), MIN_TTL=0, and so on in the original CSV. Project preprocessing only fills in NaN/ instead of finite values, and the limited 0 will remain the same, instead of filling in the missing value, turning TTL into 0. There is no silent change in the zero value.

** is still uncertain as to what causes the upstream **: whether these 0 represent the original network packets, the special code of the transmitter, the absence of the / unobserved state, or data set generation processing, which currently does not have the original PCAP, the transmitter version / configuration to distinguish from the extraction log. It is not possible to judge the faulty data from CSV to 0 directly, nor can it be considered as a confirmed real network TTL. If this level needs to be traced, the evidence of the original extraction source for this batch of CSV correspondents should be supplemented; The data set of the same name on the web is no longer used as a reference to the fact that the source is local.

## 5. Re-record and share entities: This round realises the possibilities

### 5.1 49 dimensional features vector reappears in training

Find the candidate first with hash, and then confirm the 49 row by element, and the candidate is inconsistent with 0:

- source calibration 45,252/360,000 = 12.5700% is exactly the same as training at least one 49 dimensional features vector.
The - source test 28,401/235,424 ≈ 12.0638% is the same as the training features vector.
The - matches the labels found; No verification of the elements with the same shape or the same hash.

This is not a complete original line repeat: SHA256 checks of all CSV fields such as time of retention, IP and labels show that the calibration / test does not have the exact same complete CSV record as the training. There are nevertheless duplicate complete records **within** partitions: training contains 1420000 rows but only 1414615 unique full records (5385 excess duplicates); calibration 3818 points, test 2266 points repeat surplus. They are derived from the existing original CSV and the original partition is not weighted according to its full content; This time they are not deleted.

### 5.2 Shared entities

Extract the total quantity of IP from the original CSV source: IP for each 40 for the training source /; calibration source 35/ for 37 and test source 37/ for 38, all of which have appeared in training correspondence roles. calibration, the source of all recorded IP and the purpose of IP testing are from entities that have been seen during these training periods.

This is a shared environment that has been confirmed by the / time distribution, not physical independent verification. The original line identity is still not crossed, and IP does not directly enter the 49 model features; Shared-entity filters cannot automatically translate into leaks that violate the freeze protocol. However, it limits the range of generalizations that a source test can describe, and may maintain the source domain fissile structure through indirect features such as TTL.

### 5.3 Repeat isn’t the whole reason

The fixed q below describes repeating / without repeating the subgroup, without recalculating another q: calibration Benign with the untrained repeated source, median as a percentage is still only `2.744138e-8`, and 284804 as a percentage is covered by 277076. In q, 38470 is ties, 27170 is repeated with features training, 11300 is not repeated. Repeats contribute a large amount of ties, but not a very small percentage or the only source of ties.

More caution is needed: the same 4456 source as features training calibration Attack all Raw is correct, but none meets the current q; This is also true for the same group 2577 as the source test. Duplication therefore cannot simply be described as making all Attack records confidently accepted. The overall mixed thresholds and the difference in scores for each class remain critical.

## 6. which questions have answers and which cannot be recovered from existing files

| Question | Current Answer | Scope of evidence |
|---|---|---|
| q Very small direct source | has been determined: mixed scores are in 38470 positive ties | total calibration scores, freeze q |
| Small probability how RF produces | has been identified specific examples: 294 pure positive +6 extremely low attack leaf average | freeze original tree, tested sample |
| Why is maximum ToN p insufficient? | The maximum-probability instance is explained by 296 pure-attack, 2 pure-benign, and 2 mixed leaves | Full-cache maximum and frozen tree paths |
| TTL/ encoding is incorrectly spelled | not detected; Four-column CSV total quantity unchanged | two-dimensional 119541696 element |
| repeats / sharing entity | does have 49 dimensional features repeat, full record repeat in partition, source entity all share | source; is not equal to the full original line over partition repeating |
| whether all non-sampled 49 column / cache inferences are correct | is not yet fully proof of | the old 53813 primitive features / primitive GPU inference; This is a four-wheeled expansion of the total |.
| Are TTL and related features the sole cause of all failures? | Not established | Distribution, training importance, and path evidence; causal dissolved in |
| 0 in the original CSV from where | is currently uncertain to the PCAP/ emitter layer | missing the original extraction source record |
Whether | has been recorded as an unrecorded human-selected model / operation | cannot be excluded by the repulsion of existing products on | the historical record gap on the wheel still retains |

## 7. Study of judgement and stop

Best-supported chain: **easy source feature–label relationships/shared partition environments → near-saturated original RF source probabilities → mixed source calibration/large ties yield tiny q → different ToN features produce tree disagreement → almost complete rejection and no accepted Attack.**

Rank→q,leaf→probability and probability→set are direct computations. Shared environment/feature patterns causing generalization failure is supported but incompletely identified. No evidence attributes the negative result to project code errors or final-bit floating-point differences.

NO_GO_EVIDENCE continues to apply only to freeze RF/ToN/Variant A/Strict-0/epsilon=.1. New data-dependent evidence should be added to the explanation of subsequent studies, but not to rewrite old reports, delete repetitions, remove TTL, retrain or adjust q to pursue approval. If research decisions are to explore improvements, it is a new work of alternative agreements, not an automatic extension of this round.

## Delivery and Returns

- `calibration_rank_decomposition.csv`: q ranking and two types of contributions.
- `frozen_model_feature_importance.csv`, `sampled_tree_path_diagnostics.csv`: freeze model structure and sampling path.
- `representative_tree_mechanisms.csv`, `representative_leaf_paths.csv`: The specific composition and branching of the recorded leaves.
- `feature_top_values_by_class.csv`, `ttl_pair_probability_counts.csv`: The total features distribution and the probability aggregation of the TTL conditions.
- `full_raw_four_feature_verification.csv`: The four columns of the full-volume closure of this extended inspection.
- `source_exact_feature_reuse.csv`, `source_full_csv_record_reuse.csv`, `source_entity_overlap.csv`, `feature_reuse_fixed_q_subgroups.csv`: three distinct dependencies, clearly separated.
- `validation.json` , `source_hashes.csv`: range, limitation and protection of freeze input.
- in this directory Python script, commands.sh and log: write this directory only and do not use the old writing experiment script.

Old products remain unchanged; Week1 hasn’t been activated, no messages have been sent to the professors.
'''
# Correct numerical rendering from computed exact quantities.
report=report.replace('119541696',str(4*(2365424+27520000))).replace('12.0638%',f'{28401/235424*100:.4f}%')
(O/'root_cause_report.md').write_text(report)
# Explicit expansion trigger log.
dump('expanded_checks.json',[{'trigger':'Frozen RF TTL importance51.64%; ToN large zero TTL fraction','check':'Full original CSV four-column reconstruction for both domains','result':'0 element mismatches; zeros exist upstream in CSV','stop':'No column-write error; PCAP/exporter provenance unavailable, do not invent cause'},{'trigger':'User requests concrete duplicate/entity causes, earlier audit not verified','check':'Full source exact49 feature match; raw full-record SHA256; original CSV IP membership','result':'12.57% calibration feature reuse; shared source entities; no cross-partition full record match; within-partition duplicates exist','stop':'Describe dependence, no dedup/retrain/new q'},{'trigger':'Source tiny p and ToN maximum below singleton requirement','check':'Original frozen tree read-only export and 53813 known audit rows traversal','result':'Source near-pure leaves and target leaf disagreement directly visible','stop':'No counterfactual feature edits, no alternate detector or new evaluation'}])
for p in O.glob('*.py'):ast.parse(p.read_text())
print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)
