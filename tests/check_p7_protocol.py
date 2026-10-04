"""Synthetic and previously exposed development checks; never evaluates heldout labels."""
import copy
import json
import tempfile
import unittest
import sys
import shutil
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cupy as cp
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from nids.coordination.evidence import Development, VisibleEvidence
from nids.coordination.replay import trajectory
from nids.data.partitions import PreparedData
from nids.evaluation.outer_data import Heldout, checked_windows
from nids.evaluation.outer_policy import DECISION_DTYPE, frozen_decisions, block_summary, paired_difference
from nids.p7 import method_kind, preflight, predict_mixtures, run_p7

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/execution_v2.json').read_text())


class ProtocolChecks(unittest.TestCase):
    def test_no_acceptance_is_not_zero_risk(self):
        ledger = np.zeros(4, dtype=DECISION_DTYPE)
        ledger['calls'] = 3; ledger['cost'] = 7
        summary, rep, blocks = block_summary(ledger, np.array([0, 1, 0, 1]),
                                              np.array([[1, 1], [2, 0], [0, 2]]), block_rows=2, minimum_blocks=2)
        self.assertEqual(summary['coverage'], 0)
        self.assertIsNone(summary['risk']); self.assertIsNone(summary['CI95']['risk'])
        self.assertEqual(summary['undefined_risk_replicates'], 3)
        self.assertEqual(summary['truth_action_matrix'], [[0, 0, 2], [0, 0, 2]])
        self.assertIsNone(paired_difference(rep, rep)['risk'])

    def test_block_resampling_matches_explicit_records(self):
        ledger = np.zeros(5, dtype=DECISION_DTYPE)
        ledger['accepted'] = [1, 0, 1, 1, 1]
        ledger['probability'] = [.8, .1, .2, .9, .1]
        ledger['calls'] = [1, 2, 3, 2, 1]; ledger['cost'] = [2, 4, 6, 4, 2]
        labels = np.array([1, 0, 1, 1, 0])
        mult = np.array([[1, 1, 1], [0, 2, 1], [2, 0, 1]])
        summary, rep, _ = block_summary(ledger, labels, mult, block_rows=2, minimum_blocks=3)
        for i, row in enumerate(mult):
            idx = np.concatenate([np.tile(np.arange(b*2, min(b*2+2, 5)), count) for b, count in enumerate(row)])
            accepted = ledger['accepted'][idx]; error = (ledger['probability'][idx] >= .5) != labels[idx]
            expected = [accepted.mean(), (accepted & error).sum()/accepted.sum(),
                        ledger['calls'][idx].mean(), ledger['cost'][idx].mean()]
            np.testing.assert_allclose(rep[i], expected)
        self.assertEqual(summary['blocks'], 3)

    def test_frozen_null_final_retains_early_accepts(self):
        fields = dict(query=np.array([[0, -1, -1], [0, 1, 2]]),
                      reliability=np.array([[.8, .8, .8], [.2, .3, .4]]),
                      fused=np.full((2, 3), .7), calls=np.array([[1, 1, 1], [1, 2, 3]]),
                      cost=np.array([[1, 1, 1], [1, 2, 3]]), p=np.full((2, 3), .7))
        result = frozen_decisions(fields, dict(cap=3, tau_stop=.8, tau_final=None))
        np.testing.assert_array_equal(result['stage'], [0, 2])
        np.testing.assert_array_equal(result['reason'], [0, 3])
        np.testing.assert_array_equal(result['accepted'], [True, False])
        bad = copy.deepcopy(fields); bad['query'][0, 1] = 1
        with self.assertRaises(ValueError):
            frozen_decisions(bad, dict(cap=3, tau_stop=.8, tau_final=None))

    def test_heldout_role_and_causal_identity(self):
        # Reads metadata and cached inputs only; no labels or outer CRE predictions.
        prepared = PreparedData(Path(CONFIG['prepared_run']))
        with tempfile.TemporaryDirectory() as tmp:
            d = Heldout(CONFIG, prepared, 'F1', Path(tmp)/'scores')
            self.assertFalse(hasattr(d, 'labels')); self.assertFalse(hasattr(d, 'label_path'))
            self.assertEqual(d.domain, 'ToN')
            self.assertEqual(d.audit['excluded_tail'], 260)
            rows = pq.read_table(ROOT/CONFIG['outputs']['shift']/'10k/ToN_windows.parquet').to_pylist()
            changed = copy.deepcopy(rows)
            next(w for w in changed if w['scheme'] == 'heldout')['shift'] = .7
            with self.assertRaises(ValueError): checked_windows(prepared, 'ToN', changed)
            # Exercise the first all-null predecessor column with the production schema.
            d.windows = d.windows[:2]
            d.write_identities(prepared, Path(tmp)/'identity.parquet')
            table = pq.read_table(Path(tmp)/'identity.parquet')
            self.assertEqual(table.num_rows, 20000)
            self.assertEqual(table['previous_window_id'].null_count, 10000)

    def test_metadata_preflight(self):
        r = preflight(CONFIG)
        self.assertEqual(r['policy_count'], 96)
        self.assertFalse(r['outer_labels_read'])

    def test_development_replay_matches_all_frozen_points(self):
        prepared = PreparedData(Path(CONFIG['prepared_run']))
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            for fold in ['F1', 'F2']:
                source = ROOT/CONFIG['outputs']['round2']/'P6'/fold
                fitted, initial, cost, policies = [json.loads((source/name).read_text()) for name in
                    ['models/fitted.json', 'initial_detector.json', 'cost_model.json', 'frozen_policies.json']]
                d = Development(CONFIG, prepared, fold, 'CIC', 'R_select')
                d.n = 256; d.shift = d.shift[:d.n]
                for name, points in policies.items():
                    kind, order = method_kind(name)
                    for point in points:
                        path = Path(tmp)/'trace.npy'; target = np.empty(d.n, dtype=DECISION_DTYPE)
                        def sink(a, b, fields): target[a:b] = frozen_decisions(fields, point)
                        args = (d, fitted, cost, initial, kind, path, order)
                        trajectory(*args, cap=point['cap'], tau_stop=point['tau_stop'], record_sink=sink)
                        # Sink extension must preserve the existing replay exactly.
                        trajectory(*args, cap=point['cap'], tau_stop=point['tau_stop'])
                        old = np.load(path)
                        np.testing.assert_array_equal(target, frozen_decisions({k: old[k] for k in old.dtype.names}, point))
                        decisions = np.load(source/'policies'/name/f'CIC_cap{point["cap"]}_decisions.npz')
                        # Random policy prefix depends on CuPy draw shape; test its sink parity above.
                        if kind != 'random':
                            np.testing.assert_array_equal(target['stage'], decisions['stage'][:d.n])
                            np.testing.assert_array_equal(target['accepted'], decisions['accepted'][:d.n])
                            saved = np.load(source/'policies'/name/f'CIC_cap{point["cap"]}_actual_query_trace.npy', mmap_mode='r')
                            np.testing.assert_array_equal(target['query'], saved['query'][:d.n])
                            np.testing.assert_allclose(target['probability'], saved['fused'][:d.n][np.arange(d.n), target['stage']], rtol=0, atol=1e-14)
                        if fold == 'F2' and name == 'coordinator':
                            self.assertFalse(target['accepted'].any())
                        checked += 1
                del d
        self.assertEqual(checked, 96)

    def test_randomized_policy_is_label_free_and_charges_one_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); p = folder/'policies/coordinator'; p.mkdir(parents=True)
            a = np.zeros(32, dtype=DECISION_DTYPE); a['calls'] = 1; a['cost'] = 2
            b = a.copy(); b['calls'] = 3; b['cost'] = 7
            np.save(p/'cap1.npy', a); np.save(p/'cap3.npy', b)
            entry = dict(method='coordinator', calls_budget=1.5, cost_budget=5,
                         selection=dict(cap_a=1, cap_b=3, lambda_a=.6))
            (folder/'budget_matched.json').write_text(json.dumps([entry]))
            predict_mixtures(CONFIG, SimpleNamespace(n=32, domain='CIC'), folder)
            draw = np.load(folder/'mixture_0_choose_a.npy'); ledger = np.load(folder/'mixture_0.npy')
            np.testing.assert_array_equal(ledger, np.where(draw, a, b))
            np.testing.assert_array_equal(ledger['cost'], np.where(draw, 2, 7))

    def test_full_orchestration_on_synthetic_data(self):
        # Full prediction/evaluation path on forty artificial records per domain.
        # Metadata preflight has its own real-input test; no real outer labels read here.
        from nids.evaluation.outer_data import PRIMARY, CONTROL, evaluation_arrays
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = copy.deepcopy(CONFIG); config['project_root'] = str(root)
            config['bootstrap'].update(replicates=8, block_windows=1)
            (root/'configs').mkdir()
            shutil.copy2(ROOT/'configs/p7_v1.json', root/'configs/p7_v1.json')
            (root/'configs/execution_v2.json').write_text(json.dumps(config))
            store = root/config['outputs']['feature_store']; store.mkdir(parents=True)
            shift = root/config['outputs']['shift']/'10k'; shift.mkdir(parents=True)
            for domain in ['CIC', 'BoT', 'ToN']:
                (shift/f'{domain}_manifest.json').write_text(json.dumps(dict(seconds=1, unique_data_windows=100)))
                np.save(store/f'{domain}_label.npy', np.tile([0, 1], 20))
                folder = root/config['outputs']['base_predictions']/domain; folder.mkdir(parents=True)
                for index, mid in enumerate(PRIMARY+CONTROL):
                    cache = np.zeros(40, dtype=[('probability', 'f8'), ('predicted_class', 'u1'), ('u', 'f8'), ('h', 'f8')])
                    cache['probability'] = np.tile([.1, .8, .6, .3], 10)
                    if index % 2: cache['probability'] = 1-cache['probability']
                    cache['predicted_class'] = cache['probability'] >= .5
                    cache['u'] = .4; cache['h'] = .8
                    np.save(folder/f'{mid}.npy', cache)
            for fold in ['F1', 'F2', 'F3']:
                source = ROOT/CONFIG['outputs']['round1']/'P4'/fold/'models'
                target = root/config['outputs']['round1']/'P4'/fold/'models'
                shutil.copytree(source, target)
                if fold != 'F3':
                    source = ROOT/CONFIG['outputs']['round2']/'P6'/fold
                    target = root/config['outputs']['round2']/'P6'/fold; (target/'models').mkdir(parents=True)
                    for name in ['models/fitted.json', 'initial_detector.json', 'cost_model.json', 'frozen_policies.json',
                                 'budget_matched.json', 'fixed_cascade_selection.json']:
                        shutil.copy2(source/name, target/name)
            class Artificial(Heldout):
                def __init__(self, c, prepared, fold, score_folder):
                    self.domain = c['folds'][fold]['heldout']; self.fold = fold
                    self.a = 0; self.b = self.n = 40; self.shift = np.full(40, .2)
                    self.score_folder = Path(score_folder)
                    self.cache = [np.load(root/c['outputs']['base_predictions']/self.domain/f'{m}.npy') for m in PRIMARY]
                    self.pre = [json.loads((root/c['outputs']['round1']/'P4'/fold/'models'/f'{m}.json').read_text()) for m in PRIMARY]
                    self.audit = dict(domain=self.domain, partition='T_heldout', rows=40, windows=4,
                                      window_size=10, rank_start=0, rank_end_exclusive=40, excluded_tail=0, initialization_rows=10)
                def write_identities(self, prepared, path):
                    path.write_text('synthetic identities; real schema tested separately')
            output = root/'outputs/outer_v2/P7'
            checked = []
            def evaluate_after_commit(c, d):
                self.assertTrue((output/'predictions_complete.json').is_file())
                for f in ['F1', 'F2', 'F3']:
                    self.assertTrue((output/f/'round1/scores/rf_Traffic.npy').is_file())
                self.assertTrue((output/'F1/round2/mixture_1.npy').is_file())
                checked.append(d.fold)
                return evaluation_arrays(c, d)
            with patch('nids.p7.Heldout', Artificial), patch('nids.p7.preflight', return_value={'files': []}), \
                    patch('nids.p7.evaluation_arrays', side_effect=evaluate_after_commit), contextlib.redirect_stdout(io.StringIO()):
                report = run_p7(config, None)
                self.assertTrue(report.is_file())
                with self.assertRaises(FileExistsError): run_p7(config, None)
            self.assertEqual(checked, ['F1', 'F2', 'F3'])
            manifest = json.loads((output/'P7_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'COMPLETE')
            self.assertEqual(manifest['round2']['F1']['policies'], 48)
            self.assertEqual(manifest['round2']['F2']['policies'], 48)
            self.assertEqual(manifest['F3_round2'], 'NOT_ELIGIBLE')
            metrics = json.loads((output/'F2/round2/policy_metrics.json').read_text())
            self.assertIsNone(metrics['coordinator/cap3']['risk'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
