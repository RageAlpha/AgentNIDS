"""P7: apply frozen P4/P6 artifacts, then evaluate; never fit or select."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .evaluation.outer_data import Heldout, PRIMARY, CONTROL, METHODS, evaluation_arrays
from .evaluation.outer_policy import DECISION_DTYPE, frozen_decisions, block_summary, paired_difference, diagnostic_queries


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(config):
    """Metadata only. Does not read heldout labels or compute model outcomes."""
    root = Path(config['project_root'])
    p4 = root/config['outputs']['round1']/'P4'
    p6 = root/config['outputs']['round2']/'P6'
    frozen = read(p6/'round2_frozen.json')
    if frozen['status'] != 'ALL_P6_ARTIFACTS_FROZEN_FOR_P7' or frozen['eligible_folds'] != ['F1', 'F2']:
        raise ValueError('Unexpected P6 freeze or eligibility')
    if read(p6/'F3_branch.json')['status'] != 'NOT_ELIGIBLE':
        raise ValueError('F3 must retain its frozen restricted branch')
    if read(p6/'P6_validation.json')['status'] != 'PASS':
        raise ValueError('P6 validation is not PASS')
    for path in [p4/'execution_v2.json', p6/'execution_v2.json']:
        if read(path) != config:
            raise ValueError(f'Execution config differs from frozen copy: {path}')
    inputs = [root/'configs/execution_v2.json', root/'configs/p7_v1.json',
              p6/'round2_frozen.json', p6/'F3_branch.json', p6/'P6_validation.json']
    # Snapshot consumed code as well as frozen fitted inputs, without hashing.
    for folder in [root/'src/nids', Path(config['prepared_run']),
                   root/config['outputs']['base_predictions'], root/config['outputs']['shift']/'10k']:
        inputs += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for fold in ['F1', 'F2', 'F3']:
        for mid in PRIMARY:
            path = p4/fold/'models'/f'{mid}.json'
            model = read(path)
            if model['status'] != 'COMPLETE' or model['fold'] != fold or model['domains'] != config['folds'][fold]['development']:
                raise ValueError('CRE provenance mismatch')
            inputs.append(path)
        inputs.append(p4/fold/'gate_report.json')
    policy_count = 0
    for fold in ['F1', 'F2']:
        folder = p6/fold
        for name in ['models/fitted.json', 'initial_detector.json', 'cost_model.json',
                     'frozen_policies.json', 'budget_matched.json', 'fixed_cascade_selection.json']:
            inputs.append(folder/name)
        policies = read(folder/'frozen_policies.json')
        for name, points in policies.items():
            method_kind(name)
            for point in points:
                if point['cap'] not in [1, 2, 3]:
                    raise ValueError('Invalid frozen cap')
                for key in ['tau_stop', 'tau_final']:
                    if point[key] is not None and not 0 <= point[key] <= 1:
                        raise ValueError('Invalid frozen threshold')
            policy_count += len(points)
        if fold == 'F2' and any(p['tau_stop'] is not None or p['tau_final'] is not None for p in policies['coordinator']):
            raise ValueError('F2 no-feasible-threshold branch changed')
    for domain in ['CIC', 'BoT', 'ToN']:
        inputs.append(root/config['outputs']['feature_store']/f'{domain}_label.npy')
    files = []
    for p in sorted(set(inputs)):
        s = p.stat(); files.append(dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns))
    if policy_count != 96:
        raise ValueError('Expected all 96 frozen P6 operating points')
    return dict(status='PASS_METADATA_ONLY', checked_at_utc=now(), policy_count=policy_count,
                outer_labels_read=False, predictions_computed=False, files=files)


def unchanged(inventory):
    for v in inventory['files']:
        s = Path(v['path']).stat()
        if (s.st_size, s.st_mtime_ns) != (v['bytes'], v['mtime_ns']):
            raise ValueError(f'Frozen input changed: {v["path"]}')


def method_kind(name):
    if name.startswith('fixed_'):
        order = list(map(int, name[6:]))
        if sorted(order) != [0, 1, 2]:
            raise ValueError('Invalid fixed permutation')
        return 'fixed', order
    aliases = dict(learned_cascade='learned', soft_moe='soft', hard_moe='hard', best_single='best')
    kind = aliases.get(name, name)
    if kind not in ['coordinator', 'learned', 'random', 'cheapest', 'uncertainty',
                    'best', 'uniform', 'static', 'confidence', 'soft', 'hard']:
        raise ValueError(f'Unknown method: {name}')
    return kind, None


def predict_round1(data, chunk):
    import cupy as cp
    from .reliability.cre import predict_cre
    data.score_folder.mkdir(parents=True)
    for i, mid in enumerate(PRIMARY):
        score = np.lib.format.open_memmap(data.score_folder/f'{mid}.npy', mode='w+',
                                         dtype=np.dtype([(m, '<f8') for m in METHODS]), shape=(data.n,))
        for a in range(0, data.n, chunk):
            b = min(a+chunk, data.n); v = data.cache[i][data.a+a:data.a+b]
            x = cp.column_stack((cp.asarray(v['h']), cp.asarray(v['u']), cp.asarray(data.shift[a:b])))
            values = predict_cre(x, data.pre[i])
            for name in METHODS:
                score[name][a:b] = cp.asnumpy(values[name])
        score.flush()
    data.open_scores()


def predict_round2(config, data, folder, chunk):
    from .coordination.replay import trajectory
    root = Path(config['project_root']); source = root/config['outputs']['round2']/'P6'/data.fold
    fitted, initial, cost, policies = [read(source/name) for name in
        ['models/fitted.json', 'initial_detector.json', 'cost_model.json', 'frozen_policies.json']]
    shift = read(root/config['outputs']['shift']/'10k'/f'{data.domain}_manifest.json')
    cost['shift_normalized'][data.domain] = shift['seconds']*1000/(shift['unique_data_windows']*10000)/cost['normalization_ms']
    write(folder/'cost_model_applied.json', cost)
    records = []
    for name, points in policies.items():
        kind, order = method_kind(name)
        for point in points:
            path = folder/'policies'/name/f'cap{point["cap"]}.npy'
            path.parent.mkdir(parents=True, exist_ok=True)
            ledger = np.lib.format.open_memmap(path, mode='w+', dtype=DECISION_DTYPE, shape=(data.n,))
            def sink(a, b, fields):
                ledger[a:b] = frozen_decisions(fields, point)
            report = trajectory(data, fitted, cost, initial, kind, path, order,
                                chunk=chunk, cap=point['cap'], tau_stop=point['tau_stop'], record_sink=sink)
            ledger.flush()
            if report['detector_record_reads'] != int(ledger['calls'].sum()):
                raise ValueError('Actual query count mismatch')
            records.append(dict(method=name, point=point, replay=report))
            print(f'{data.fold}/{name}/cap{point["cap"]}: frozen predictions written', flush=True)
    write(folder/'prediction_manifest.json', records)
    for name in ['frozen_policies.json', 'budget_matched.json', 'fixed_cascade_selection.json']:
        shutil.copy2(source/name, folder/name)
    predict_mixtures(config, data, folder)


def predict_mixtures(config, data, folder):
    """Commit runtime random choices without labels, before any evaluation."""
    import cupy as cp
    for index, entry in enumerate(read(folder/'budget_matched.json')):
        choice = entry['selection']
        if choice is None:
            continue
        points = folder/'policies'/entry['method']
        a = np.load(points/f'cap{choice["cap_a"]}.npy', mmap_mode='r')
        b = np.load(points/f'cap{choice["cap_b"]}.npy', mmap_mode='r')
        rng = cp.random.RandomState(config['seed']+int(entry['calls_budget']*4)*10+['CIC', 'BoT', 'ToN'].index(data.domain))
        mixed = np.lib.format.open_memmap(folder/f'mixture_{index}.npy', mode='w+', dtype=DECISION_DTYPE, shape=(data.n,))
        draws = np.lib.format.open_memmap(folder/f'mixture_{index}_choose_a.npy', mode='w+', dtype=bool, shape=(data.n,))
        draws[:] = cp.asnumpy(rng.random_sample(data.n) < choice['lambda_a'])
        for start in range(0, data.n, 262144):
            end = min(start+262144, data.n)
            mixed[start:end] = np.where(draws[start:end], a[start:end], b[start:end])
        mixed.flush(); draws.flush()


def evaluate_round1(config, data, folder, rng):
    import cupy as cp
    from .evaluation.round1_metrics import probability_metrics, detection_metrics, complementarity
    from .evaluation.round1_bootstrap import domain_bootstrap
    arrays = evaluation_arrays(config, data)
    result = dict(audit=arrays['audit'], CRE={}, detectors={}, G0={})
    masks = {'all': cp.ones(data.n, bool), 'Benign': arrays['labels'] == 0, 'Attack': arrays['labels'] == 1}
    for i, mid in enumerate(PRIMARY):
        correct = (~arrays['errors'][:, i]).astype(cp.uint8)
        score = np.load(data.score_folder/f'{mid}.npy', mmap_mode='r')
        result['CRE'][mid] = {s: {m: probability_metrics(correct[mask], cp.asarray(score[m])[mask], config,
                              ranking_only=m == 'one_minus_u') for m in METHODS} for s, mask in masks.items()}
    for s, mask in masks.items():
        result['G0'][s] = dict(primary=complementarity(arrays['errors'][mask], PRIMARY),
                              All49=complementarity(arrays['control_errors'][mask], CONTROL))
        result['detectors'][s] = {m: detection_metrics(arrays['labels'][mask], arrays['probability'][m][mask], config) for m in PRIMARY+CONTROL}
    write(folder/'metrics.json', result)
    bootstrap = folder/'bootstrap'; bootstrap.mkdir()
    boot = domain_bootstrap(config, arrays, data.score_folder, bootstrap, rng)
    aggregate_round1(config, [(result, boot)], folder)
    return result, boot


def evaluate_round2(config, data, folder, multiplicity):
    import cupy as cp
    from .evaluation.round1_metrics import probability_metrics
    labels = np.load(Path(config['project_root'])/config['outputs']['feature_store']/f'{data.domain}_label.npy', mmap_mode='r')[data.a:data.b]
    policies = read(folder/'frozen_policies.json'); summaries = {}; reps = {}
    block_rows = config['bootstrap']['block_windows']*data.audit.get('window_size', 10000)
    for name, points in policies.items():
        for point in points:
            key = f'{name}/cap{point["cap"]}'; path = folder/'policies'/name/f'cap{point["cap"]}.npy'
            ledger = np.load(path, mmap_mode='r')
            result, rep, blocks = block_summary(ledger, labels, multiplicity, block_rows=block_rows,
                minimum_blocks=config['bootstrap']['minimum_units_per_evaluation_domain'])
            result['G2'] = diagnostic_queries(ledger, labels, data.cache, data.a)
            correct = (cp.asarray(ledger['probability']) >= .5) == cp.asarray(labels)
            result['reliability_metrics'] = probability_metrics(correct.astype(cp.uint8), cp.asarray(ledger['reliability']), config,
                ranking_only=name in ['uniform', 'static', 'confidence', 'soft_moe'])
            result['point'] = point
            summaries[key], reps[key] = result, rep
            np.save(path.with_suffix('.bootstrap.npy'), rep); np.save(path.with_suffix('.blocks.npy'), blocks)
            print(f'{data.fold}/{key}: outer evaluation written', flush=True)
    # All frozen mixtures, including infeasible entries, retain their original identities.
    mixtures = []; mixture_reps = {}
    for index, entry in enumerate(read(folder/'budget_matched.json')):
        choice = entry['selection']
        if choice is None:
            mixtures.append(dict(frozen=entry, outer=None)); continue
        name = entry['method']
        mixture_path = folder/f'mixture_{index}.npy'
        mixed = np.load(mixture_path, mmap_mode='r')
        result, rep, blocks = block_summary(mixed, labels, multiplicity, block_rows=block_rows,
            minimum_blocks=config['bootstrap']['minimum_units_per_evaluation_domain'])
        result['calls_budget_met_on_outer'] = result['mean_calls'] <= entry['calls_budget']+1e-12
        result['cost_budget_met_on_outer'] = result['normalized_cost'] <= entry['cost_budget']+1e-12
        expected = {k: choice['lambda_a']*summaries[f'{name}/cap{choice["cap_a"]}'][k]
                    +(1-choice['lambda_a'])*summaries[f'{name}/cap{choice["cap_b"]}'][k]
                    for k in ['coverage', 'accepted_error_mass', 'mean_calls', 'normalized_cost']}
        expected['risk'] = expected['accepted_error_mass']/expected['coverage'] if expected['coverage'] else None
        mixtures.append(dict(frozen=entry, outer=result, outer_expected=expected,
                             ledger=str(mixture_path), choices=str(folder/f'mixture_{index}_choose_a.npy')))
        mixture_reps[(name, entry['calls_budget'])] = rep
        np.save(folder/f'mixture_{index}.bootstrap.npy', rep); np.save(folder/f'mixture_{index}.blocks.npy', blocks)
    comparisons = []
    for (name, budget), other in mixture_reps.items():
        key = ('coordinator', budget)
        if name != 'coordinator' and key in mixture_reps:
            estimable = len(multiplicity[0]) >= config['bootstrap']['minimum_units_per_evaluation_domain']
            comparisons.append(dict(baseline=name, calls_budget=budget, estimable=estimable,
                difference='coordinator_minus_baseline', CI95=paired_difference(mixture_reps[key], other) if estimable else None,
                interpretation='paired_descriptive_pointwise; verify realized budgets; no outer reselection'))
    fixed = read(folder/'fixed_cascade_selection.json')
    fixed_summary = {f'cap{cap}': dict(selected_method=name, result=summaries[f'{name}/cap{cap}']) for cap, name in fixed.items()}
    write(folder/'policy_metrics.json', summaries); write(folder/'budget_metrics.json', mixtures)
    write(folder/'paired_budget_comparisons.json', comparisons); write(folder/'selected_fixed_cascade.json', fixed_summary)
    return dict(policies=len(summaries), budget_entries=len(mixtures), feasible_budget_entries=len(mixture_reps))


def aggregate_round1(config, evaluations, output):
    from .evaluation.round1_bootstrap import percentile_ci, gate
    g0 = np.mean([v[1][0] for v in evaluations], axis=0)
    aurc = np.mean([v[1][1] for v in evaluations], axis=0)
    g1 = (aurc[:, :, METHODS.index('hu')]-aurc[:, :, METHODS.index('hus')]).mean(axis=1)
    points = dict(G0=float(np.mean([v[0]['G0']['all']['primary']['oracle_headroom'] for v in evaluations])),
                  G1=float(np.mean([v[0]['CRE'][m]['all']['hu']['AURC']-v[0]['CRE'][m]['all']['hus']['AURC'] for v in evaluations for m in PRIMARY])))
    estimable = all(v[1][2]['blocks'] >= config['bootstrap']['minimum_units_per_evaluation_domain'] for v in evaluations)
    gates = {}
    for name, rep in [('G0', g0[:, 0]), ('G1', g1)]:
        interval = percentile_ci(rep); delta = config['gates'][name]['minimum_effect']
        gates[name] = dict(estimate=points[name], CI95=interval, minimum_effect=delta,
                           status=gate(points[name], interval, delta, estimable), estimable=estimable)
    np.save(output/'outer_G0.npy', g0); np.save(output/'outer_AURC.npy', aurc); np.save(output/'outer_G1.npy', g1)
    detectors = {}
    for i, mid in enumerate(PRIMARY):
        methods = {m: dict(AURC=float(np.mean([v[0]['CRE'][mid]['all'][m]['AURC'] for v in evaluations])),
                           CI95=percentile_ci(aurc[:, i, j])) for j, m in enumerate(METHODS)}
        detectors[mid] = dict(methods=methods, delta_hu_minus_hus=dict(
            estimate=methods['hu']['AURC']-methods['hus']['AURC'],
            CI95=percentile_ci(aurc[:, i, METHODS.index('hu')]-aurc[:, i, METHODS.index('hus')])))
    write(output/'round1_aggregate.json', dict(gates=gates, model_AURC=detectors,
          All49_G0_CI95=percentile_ci(g0[:, 1]), domains=[v[0]['audit']['domain'] for v in evaluations],
          weighting='evaluated heldout domains equally; three detectors equally for G1',
          uncertainty_scope=config['bootstrap']['uncertainty_scope'], round2_eligibility='unchanged_from_P4_development'))
    return gates


def run_p7(config, prepared):
    import cupy as cp
    import torch
    from .p4 import release
    from .models.base_models import GpuMemoryMonitor
    root = Path(config['project_root']); rules = read(root/'configs/p7_v1.json')
    inventory = preflight(config)
    output = root/rules['output']
    if output.exists():
        raise FileExistsError(f'P7 output already exists; preserve it and audit before any rerun: {output}')
    output.mkdir(parents=True)
    write(output/'upstream_inventory.json', inventory)
    shutil.copy2(root/'configs/p7_v1.json', output/'p7_rules.json')
    shutil.copy2(root/'configs/execution_v2.json', output/'execution_v2.json')
    started = time.perf_counter(); monitor = GpuMemoryMonitor(0); monitor.start()
    try:
        data = []
        for fold in ['F1', 'F2', 'F3']:
            d = Heldout(config, prepared, fold, output/fold/'round1/scores')
            data.append(d)
        # Conservative on-disk upper estimate includes all feasible mixture ledgers.
        estimate = sum(d.n*(3*len(METHODS)*8+128) for d in data)
        for d in data[:2]:
            source = root/config['outputs']['round2']/'P6'/d.fold
            points = sum(map(len, read(source/'frozen_policies.json').values()))
            mixtures = sum(e['selection'] is not None for e in read(source/'budget_matched.json'))
            estimate += d.n*(points*DECISION_DTYPE.itemsize+mixtures*(DECISION_DTYPE.itemsize+1))
        if shutil.disk_usage(output).free < estimate*1.1:
            raise OSError(f'Insufficient space: estimated P7 artifacts need {estimate*1.1/1e9:.1f} GB free')
        write(output/'status.json', dict(status='PREDICTING', started_at_utc=now(), outer_labels_read=False,
                                         estimated_artifact_bytes=estimate))
        for d in data:
            folder = output/d.fold; folder.mkdir(exist_ok=True)
            d.write_identities(prepared, folder/'identities.parquet'); write(folder/'data_audit.json', d.audit)
            predict_round1(d, config['batches']['inference_rows'])
            if d.fold in ['F1', 'F2']:
                predict_round2(config, d, folder/'round2', rules['replay_chunk_rows'])
            else:
                write(folder/'round2_branch.json', dict(status='NOT_ELIGIBLE', source='P4 development G1 INCONCLUSIVE', executed=False))
            release()
        unchanged(inventory)
        write(output/'predictions_complete.json', dict(status='ALL_FOLDS_PREDICTIONS_COMMITTED', completed_at_utc=now(),
                                                       labels_read_by_prediction=False, folds=['F1', 'F2', 'F3']))
        # This explicit transition precedes the first heldout label read.
        write(output/'status.json', dict(status='EVALUATING', outer_label_evaluation_started_at_utc=now()))
        rng = cp.random.RandomState(config['seed']); evaluations = []; round2 = {}
        for d in data:
            folder = output/d.fold/'round1'
            evaluations.append(evaluate_round1(config, d, folder, rng))
            release()
            if d.fold in ['F1', 'F2']:
                multiplicity = np.load(folder/'bootstrap'/f'{d.domain}_block_multiplicity.npy')
                round2[d.fold] = evaluate_round2(config, d, output/d.fold/'round2', multiplicity)
                release()
        gates = aggregate_round1(config, evaluations, output)
        unchanged(inventory)
        result = dict(status='COMPLETE', completed_at_utc=now(), seconds=time.perf_counter()-started,
                      round1_gates=gates, round2=round2, F3_round2='NOT_ELIGIBLE', upstream_unchanged=True,
                      gpu_memory=monitor.stop(), gpu=torch.cuda.get_device_name(0), p7_rules=rules['version'])
        write(output/'P7_manifest.json', result); write(output/'status.json', result)
        lines = ['# P7 outer evaluation report', '', 'State: COMPLETE', '',
                 'The old P4/P6 rule is fixed with the development of the selection value; Three-fold predicts the associated outer layer labels after all has been preserved.', '',
                 '| threshold | range estimates | 95% CI | |', '|---|---:|---|---|']
        for name, v in gates.items():
            lines.append(f'| {name} | {v["estimate"]:.8f} | {v["CI95"]} | {v["status"]} |')
        lines += ['', 'Round2 is only F1/F2, with policy_metrics and budget_metrics remaining complete. F2 freeze strategy no feasible threshold, not re-selected; F3 is not fully coordinated.',
                  '', 'Development budgets do not guarantee that the actual external cost meets the same budget and exceeds individual labels. zero acceptance risk undefined; The range is conditioned by existing models and evaluation domains, with comparative descriptive evidence as a point.',
                  '', 'This report does not automatically declare coordination superior to baseline nor does it modify the rules of Item3. P5 sensitivity expansion and P8 are not within the scope of this P7-v1 implementation.', '']
        (output/'stage_report.md').write_text('\n'.join(lines))
        return output/'stage_report.md'
    except Exception as error:
        # Preserve completed artifacts and exposure status; never silently restart.
        prior = read(output/'status.json') if (output/'status.json').exists() else {}
        write(output/'failure.json', dict(status='INCOMPLETE', at_utc=now(), error_type=type(error).__name__,
                                         error=str(error), prior_status=prior, gpu_memory=monitor.stop()))
        raise
