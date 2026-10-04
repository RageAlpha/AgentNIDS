import numpy as np

def ratio(a, b):
    return a / b if b else None

def sets(p, qb, qa):
    return (p <= qb, 1.0 - p <= qa)

def counts(p, y, qb, qa):
    b, a = sets(p, qb, qa)
    bs = b & ~a
    ats = a & ~b
    single = b ^ a
    c = dict(n=len(p), true_benign=np.count_nonzero(y == 0), true_attack=np.count_nonzero(y == 1), benign_singleton=bs.sum(), attack_singleton=ats.sum(), empty=(~b & ~a).sum(), double=(b & a).sum(), tn=(bs & (y == 0)).sum(), fn=(bs & (y == 1)).sum(), fp=(ats & (y == 0)).sum(), tp=(ats & (y == 1)).sum(), accepted=single.sum(), singleton_errors=(bs & (y == 1) | ats & (y == 0)).sum(), raw_errors=((p >= 0.5) != y).sum(), singleton_raw_disagreement=(single & (a != (p >= 0.5))).sum(), benign_covered=(b & (y == 0)).sum(), attack_covered=(a & (y == 1)).sum())
    c = {k: int(v) for k, v in c.items()}
    assert c['benign_singleton'] + c['attack_singleton'] + c['empty'] + c['double'] == c['n']
    assert c['tn'] + c['fn'] + c['fp'] + c['tp'] == c['accepted']
    assert c['fn'] + c['fp'] == c['singleton_errors']
    return c

def unit_checks():
    cases = [([0, 0.25, 0.5, 0.75, 1], 0.25, 0.25, [1, 1, 0, 0, 0], [0, 0, 0, 1, 1]), ([0.125, 0.25, 0.5, 0.75, 0.875], 0.75, 0.75, [1, 1, 1, 1, 0], [0, 1, 1, 1, 1]), ([0.5], 0.5, 0.5, [1], [1]), ([0.25, 0.375, 0.5, 0.75], 0.25, 0.625, [1, 0, 0, 0], [0, 1, 1, 1])]
    for p, qb, qa, eb, ea in cases:
        b, a = sets(np.array(p), qb, qa)
        assert b.tolist() == [bool(x) for x in eb]
        assert a.tolist() == [bool(x) for x in ea]
    c = counts(np.array([0.375]), np.array([1]), 0.25, 0.625)
    assert c['tp'] == 1 and c['raw_errors'] == 1 and (c['singleton_errors'] == 0) and (c['singleton_raw_disagreement'] == 1)
    for qb, qa in [(0.25, 0.25), (0.75, 0.75)]:
        c = counts(np.array([0.5]), np.array([1]), qb, qa)
        assert c['accepted'] == 0 and ratio(c['singleton_errors'], c['accepted']) is None
    return dict(inclusive_equality=True, asymmetric_thresholds=True, empty=True, double=True, zero_acceptance_NA=True, singleton_can_differ_from_raw=True, label_free_set_function=True)
