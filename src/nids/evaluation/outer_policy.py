"""Frozen policy application and paired outer summaries; no threshold selection."""
from __future__ import annotations

import numpy as np

DECISION_DTYPE = np.dtype([
    ('stage', 'u1'), ('accepted', '?'), ('reason', 'u1'),
    ('probability', '<f8'), ('reliability', '<f8'), ('calls', 'u1'), ('cost', '<f8'),
    ('query', 'i1', (3,)), ('query_probability', '<f8', (3,)),
])


def frozen_decisions(fields, point):
    """Use P6's logical terminal stage, including hard-MoE's reused final slot."""
    cap, tau_stop, tau_final = point['cap'], point['tau_stop'], point['tau_final']
    n = len(fields['query'])
    crossing = fields['reliability'][:, :cap-1] >= tau_stop if tau_stop is not None else np.zeros((n, cap-1), bool)
    early = crossing.any(axis=1)
    stage = np.where(early, crossing.argmax(axis=1), cap-1) if cap > 1 else np.zeros(n, int)
    row = np.arange(n)
    r = fields['reliability'][row, stage]
    accepted = early | ((~early) & (r >= tau_final)) if tau_final is not None else early
    out = np.empty(n, dtype=DECISION_DTYPE)
    out['stage'], out['accepted'] = stage, accepted
    out['reason'] = np.where(early, 0, np.where(accepted, 1, 2 if tau_final is not None else 3))
    out['probability'], out['reliability'] = fields['fused'][row, stage], r
    out['calls'], out['cost'] = fields['calls'][row, stage], fields['cost'][row, stage]
    out['query'], out['query_probability'] = fields['query'], fields['p']
    for step in range(3):
        if np.any(out['query'][stage < step, step] >= 0):
            raise ValueError('Detector was queried after the decision')
        out['query_probability'][out['query'][:, step] < 0, step] = 0
    if not np.array_equal(out['calls'], (out['query'] >= 0).sum(axis=1)):
        raise ValueError('Charged calls disagree with actual queries')
    return out


def ci(values):
    """A zero-acceptance bootstrap draw has undefined risk, not zero risk."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        return None
    return np.percentile(values, [2.5, 97.5]).tolist()


def block_summary(ledger, labels, multiplicity, block_rows=100000, minimum_blocks=20):
    import cupy as cp
    n = len(ledger)
    blocks = (n+block_rows-1)//block_rows
    stats = cp.zeros((blocks, 5), dtype=cp.float64)
    matrix = cp.zeros((2, 3), dtype=cp.int64)
    for a in range(0, n, block_rows):
        b = min(a+block_rows, n)
        v = ledger[a:b]
        y = cp.asarray(labels[a:b]); accepted = cp.asarray(v['accepted'])
        pred = cp.asarray(v['probability']) >= .5
        stats[a//block_rows] = cp.asarray([b-a, int(accepted.sum()), int((accepted & (pred != y)).sum()),
                                           float(cp.asarray(v['calls']).sum()), float(cp.asarray(v['cost']).sum())])
        action = cp.where(accepted, pred.astype(cp.int64), 2)
        matrix += cp.bincount(y.astype(cp.int64)*3+action, minlength=6).reshape(2, 3)
    totals = stats.sum(axis=0)
    point = dict(coverage=float(totals[1]/totals[0]), risk=float(totals[2]/totals[1]) if totals[1] else None,
                 accepted_error_mass=float(totals[2]/totals[0]), mean_calls=float(totals[3]/totals[0]),
                 normalized_cost=float(totals[4]/totals[0]))
    draws = cp.asarray(multiplicity) @ stats
    rep = cp.column_stack((draws[:, 1]/draws[:, 0],
                           cp.where(draws[:, 1] > 0, draws[:, 2]/cp.maximum(draws[:, 1], 1), cp.nan),
                           draws[:, 3]/draws[:, 0], draws[:, 4]/draws[:, 0]))
    rep = cp.asnumpy(rep)
    names = ['coverage', 'risk', 'mean_calls', 'normalized_cost']
    matrix = cp.asnumpy(matrix)
    strata = {}
    for label, name in enumerate(['Benign', 'Attack']):
        count = int(matrix[label].sum()); accepted = int(matrix[label, :2].sum())
        strata[name] = dict(rows=count, coverage=accepted/count if count else None,
                            selective_risk=int(matrix[label, 1-label])/accepted if accepted else None,
                            abstention_rate=int(matrix[label, 2])/count if count else None)
    return dict(**point, rows=n, blocks=blocks, estimable=blocks >= minimum_blocks,
                CI95={k: ci(rep[:, j]) if blocks >= minimum_blocks else None for j, k in enumerate(names)},
                undefined_risk_replicates=int((~np.isfinite(rep[:, 1])).sum()),
                truth_action_matrix=matrix.tolist(), truth_order=['Benign', 'Attack'],
                action_order=['Benign', 'Attack', 'Abstain'], strata=strata), rep, cp.asnumpy(stats)


def paired_difference(a, b):
    names = ['coverage', 'risk', 'mean_calls', 'normalized_cost']
    return {key: ci(a[:, i]-b[:, i]) for i, key in enumerate(names)}


def diagnostic_queries(ledger, labels, caches, offset=0, chunk=262144):
    import cupy as cp
    counts = np.zeros((2, 4), dtype=np.int64)
    for a in range(0, len(ledger), chunk):
        b = min(a+chunk, len(ledger)); v = ledger[a:b]
        y = cp.asarray(labels[a:b]); query = cp.asarray(v['query']); p = cp.asarray(v['query_probability'])
        correct = cp.column_stack([cp.asarray(c['predicted_class'][offset+a:offset+b]) == y for c in caches])
        seen = cp.zeros_like(correct); rows = cp.arange(b-a)
        for step in range(3):
            executed = query[:, step] >= 0
            if step:
                wrong = (p[:, step-1] >= .5) != y
                new_correct = (p[:, step] >= .5) == y
                opportunity = wrong & (correct & ~seen).any(axis=1)
                counts[step-1] += [int(executed.sum()), int((executed & wrong & new_correct).sum()),
                                   int((executed & opportunity).sum()), int((executed & ~wrong & ~new_correct).sum())]
            seen[rows[executed], query[executed, step]] = True
    return [dict(step=i+2, consulted=int(c[0]), corrective_hits=int(c[1]), oracle_opportunities=int(c[2]),
                 consultation_harms=int(c[3]), corrective_yield=c[1]/c[0] if c[0] else None,
                 opportunity_hit_rate=c[1]/c[2] if c[2] else None,
                 consultation_harm=c[3]/c[0] if c[0] else None) for i, c in enumerate(counts)]
