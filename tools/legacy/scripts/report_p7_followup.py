"""Read frozen results; export reporting diagnostics without fitting or selection."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/p7_report_followup'
PRIMARY = ['rf_Traffic', 'xgb_Protocol', 'mlp_Behavior']
METHODS = ['prior', 'raw_h', 'one_minus_u', 'u', 'hu', 'hus']


def read(path):
    return json.loads((ROOT / path).read_text())


def save_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    ranking, distribution, reliability, detectors, comparisons, policies = [], [], [], [], [], []
    with (ROOT/'outputs/round1_data_prep_v1/split_manifest.csv').open() as f:
        for r in csv.DictReader(f):
            if r['domain']=='UNSW' and r['scheme']=='source':
                n, b, a = (int(r[k]) for k in ['rows','benign','attack'])
                assert n==b+a
                distribution.append(dict(fold='shared_source',partition=r['partition'],domain='UNSW',
                    rows=n,benign=b,attack=a,attack_fraction=a/n))
    for fold in ['F1', 'F2', 'F3']:
        base = f'outputs/outer_v2/P7/{fold}/round1'
        outer = read(f'{base}/metrics.json')
        domain = outer['audit']['domain']
        with (ROOT / f'outputs/round1_v2/P4/{fold}/CRE_metrics.csv').open() as f:
            dev = list(csv.DictReader(f))
        audits = read(f'outputs/round1_v2/P4/{fold}/manifest.json')['training_audits']
        for a in audits + [outer['audit']]:
            distribution.append(dict(fold=fold, partition=a['partition'], domain=a['domain'],
                rows=a['rows'], benign=a['benign'], attack=a['attack'], attack_fraction=a['attack']/a['rows']))
        for part in ['R_cal', 'R_select']:
            for dom in sorted({r['domain'] for r in dev}):
                rs = {r['stratum']: r for r in dev if r['partition']==part and r['domain']==dom
                      and r['detector']==PRIMARY[0] and r['method']=='prior'}
                n, b, a = (int(rs[s]['rows']) for s in ['all', 'Benign', 'Attack'])
                assert n == b+a
                distribution.append(dict(fold=fold, partition=part, domain=dom, rows=n,
                                         benign=b, attack=a, attack_fraction=a/n))
        for model in PRIMARY:
            fitted = read(f'outputs/round1_v2/P4/{fold}/models/{model}.json')
            coefficient = fitted['models']['u']['coefficients'][0]
            assert fitted['models']['u']['indices'] == [1] and fitted['scale'][1] > 0
            scores = np.load(ROOT / f'{base}/scores/{model}.npy', mmap_mode='r')
            ix = np.linspace(0, len(scores)-1, min(20000, len(scores)), dtype=np.int64)
            sample = scores[ix]
            raw_entropy = 1-sample['one_minus_u']
            x = (raw_entropy-fitted['anchor'][1]-fitted['mean_offset'][1])/fitted['scale'][1]
            logits = coefficient*x+fitted['models']['u']['intercept']
            reconstructed = 1/(1+np.exp(-logits))
            error = float(np.max(np.abs(reconstructed-sample['u'])))
            order = np.argsort(raw_entropy)
            differences = np.diff(sample['u'][order]) * np.sign(coefficient)
            # Reconstructing u from 1-u may merge floating-point neighbours.
            violations = int((differences < -1e-12).sum())
            assert error < 1e-12 and violations == 0
            z = outer['CRE'][model]['all']
            delta = z['u']['AURC']-z['one_minus_u']['AURC']
            if coefficient < 0:
                assert abs(delta) < 1e-10
            ranking.append(dict(fold=fold, domain=domain, detector=model, coefficient_standardized_u=coefficient,
                entropy_scale=fitted['scale'][1], direction_vs_one_minus_u='opposite' if coefficient>0 else 'same',
                AURC_one_minus_u=z['one_minus_u']['AURC'], AURC_f_u=z['u']['AURC'], delta_f_u_minus_confidence=delta,
                ties_one_minus_u=z['one_minus_u']['score_tie_groups'], ties_f_u=z['u']['score_tie_groups'],
                sampled_records=len(ix), sample_max_prediction_error=error, sample_monotonicity_violations=violations,
                model_source=f'outputs/round1_v2/P4/{fold}/models/{model}.json', metric_source=f'{base}/metrics.json'))
            for stratum in ['all', 'Benign', 'Attack']:
                for method in METHODS:
                    z = outer['CRE'][model][stratum][method]
                    reliability.append(dict(fold=fold, domain=domain, detector=model, stratum=stratum, method=method,
                        rows=z['rows'], correct=z['positive'], incorrect=z['negative'],
                        **{k:z.get(k) for k in ['AURC', 'AUROC', 'Brier', 'NLL']}))
        for model,z in outer['detectors']['all'].items():
            detectors.append(dict(fold=fold, domain=domain, detector=model, rows=z['rows'],
                **{k:z[k] for k in ['TP','FP','FN','TN','accuracy','balanced_accuracy','recall']},
                FPR=z['FP']/(z['FP']+z['TN'])))
        for method in METHODS:
            for metric in ['AURC','Brier','NLL']:
                if method=='one_minus_u' and metric!='AURC':
                    continue
                selected = [r for r in dev if r['partition']=='R_select' and r['stratum']=='all' and r['method']==method]
                assert len(selected)==6
                development = sum(float(r[metric]) for r in selected)/6
                target = sum(outer['CRE'][m]['all'][method][metric] for m in PRIMARY)/3
                comparisons.append(dict(fold=fold, target=domain, method=method, metric=metric,
                    development_R_select_equal_domains_detectors=development,
                    outer_equal_detectors=target, outer_minus_development=target-development))
        if fold!='F3':
            frozen = read(f'outputs/round2_v2/P6/{fold}/frozen_policies.json')
            outer_policy = read(f'outputs/outer_v2/P7/{fold}/round2/policy_metrics.json')
            for method,cap in [('coordinator',3),('fixed_102',3),('hard_moe',2)]:
                point = next(r for r in frozen[method] if r['cap']==cap)
                for stage,z in [('development_R_select',point['summary']),('outer',outer_policy[f'{method}/cap{cap}'])]:
                    policies.append(dict(fold=fold,target=domain,method=method,cap=cap,stage=stage,
                        **{k:z[k] for k in ['coverage','risk','mean_calls','normalized_cost']}))
    save_csv('entropy_order_audit.csv', ranking)
    save_csv('class_distribution.csv', distribution)
    save_csv('outer_stratified_reliability.csv', reliability)
    save_csv('outer_detector_classification.csv', detectors)
    save_csv('development_outer_reliability.csv', comparisons)
    save_csv('development_outer_policies.csv', policies)
    result = dict(status='PASS_READ_ONLY_DIAGNOSTIC', scope='frozen coefficients and saved metrics; deterministic 20k score samples per fold-detector, not full-row reranking',
        sampled_records=sum(r['sampled_records'] for r in ranking),
        opposite_order_models=sum(r['direction_vs_one_minus_u']=='opposite' for r in ranking),
        same_order_models=sum(r['direction_vs_one_minus_u']=='same' for r in ranking),
        maximum_sample_prediction_error=max(r['sample_max_prediction_error'] for r in ranking),
        sample_monotonicity_violations=sum(r['sample_monotonicity_violations'] for r in ranking),
        outer_records=sum(r['rows'] for r in distribution if r['partition']=='T_heldout'),
        no_fitting_or_selection=True, source_code=['src/nids/reliability/cre.py','src/nids/evaluation/round1_metrics.py'])
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
