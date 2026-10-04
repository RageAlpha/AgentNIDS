"""Week 0 draft: float64 scores, exact rational rank; synthetic tests only here."""
import math
from fractions import Fraction
import numpy as np

def checked_p(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('finite one-dimensional probabilities in [0,1] required')
    return p

def rf_p(p):
    p = checked_p(p)
    if not np.array_equal(p, p.astype(np.float32).astype(np.float64)):
        raise ValueError("RF contract: probability must be exact float32 value stored in float64; no rounding performed")
    return p

def scores(p, y):
    p = rf_p(p)
    y = np.asarray(y)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('aligned binary labels required')
    return np.where(y == 0, p, 1.0-p)

def rank_k(n, epsilon='1/10'):
    e = Fraction(epsilon)
    if n < 1 or not 0 < e < 1:
        raise ValueError('n>=1 and 0<epsilon<1 required')
    v = (n+1)*(1-e)
    return (v.numerator + v.denominator-1)//v.denominator

def threshold(a, epsilon='1/10'):
    a = checked_p(a)
    k = rank_k(len(a), epsilon)
    return (math.inf if k > len(a) else float(np.sort(a)[k-1])), k

def prediction_sets(p, q):
    p = rf_p(p)
    if math.isnan(q) or q < 0:
        raise ValueError('invalid q')
    # columns Benign, Attack; deliberately accepts no evaluation labels.
    return np.column_stack((p <= q, 1.0-p <= q))

def serialize_q(q):
    return {'kind': 'positive_infinity'} if q == math.inf else {'kind': 'float64_hex', 'value': float(q).hex()}

def restore_q(obj):
    if obj['kind'] == 'positive_infinity': return math.inf
    if obj['kind'] != 'float64_hex': raise ValueError('unknown marker')
    return float.fromhex(obj['value'])

def synthetic_tests():
    checks = {}
    rng = np.random.default_rng(20260915)
    for q in [0., .25, .5, .75, 1., math.inf]:
        p = np.r_[0., .5, 1., rng.random(1000).astype(np.float32).astype(np.float64), np.repeat(.5, 10000)]
        if math.isfinite(q): p = np.r_[p, q, 1-q, float(np.nextafter(np.float32(.5),np.float32(0))), float(np.nextafter(np.float32(.5),np.float32(1)))]
        actual = prediction_sets(p, q)
        expected = np.array([[float(x)<=q, 1.0-float(x)<=q] for x in p])
        assert np.array_equal(actual, expected)
        size = actual.sum(1)
        codes = actual[:,0].astype(int)+2*actual[:,1].astype(int)
        assert np.bincount(codes, minlength=4).sum() == len(p)
        if q < .5: assert not (size==2).any()
        if q > .5: assert not (size==0).any()
        if q == .5:
            assert (size[p==.5]==2).all() and (size[p!=.5]==1).all()
        restored = restore_q(serialize_q(q))
        assert restored == q and np.array_equal(prediction_sets(p,restored), actual)
        y = rng.integers(0,2,len(p)); y2 = 1-y
        def evaluation(p,y,q):
            c = prediction_sets(p,q)
            return c, c[np.arange(len(y)),y].mean()
        assert np.array_equal(evaluation(p,y,q)[0],evaluation(p,y2,q)[0])
    checks['endpoints_boundaries_three_structures_vector_scalar_counts_label_independence_serialization'] = 'PASS'
    for a,e,k in [([.2],'1/2',1),([.2,.8],'1/3',2),([.2,.8],'1/10',3),([.1]*10000,'1/10',9001)]:
        q, actual_k = threshold(a,e)
        assert actual_k==k and q==(math.inf if k>len(a) else sorted(a)[k-1])
    assert rank_k(360000)==324001
    assert np.array_equal(scores([0,.5,1,0,.5,1],[0,0,0,1,1,1]),[0,.5,1,1,.5,0])
    assert scores([.25],[0])[0] != scores([.25],[1])[0]
    try:
        prediction_sets([np.nextafter(.5,0)], .5)
        raise AssertionError("out-of-contract probability accepted")
    except ValueError:
        pass
    checks["reject_non_float32_origin_probability_without_rounding"]="PASS"
    for q in [.1, np.nextafter(.5,1),np.nextafter(.5,0)]:
        assert restore_q(serialize_q(q)).hex()==float(q).hex()
    checks['k_1_n_over_n_large_ties_exact_rank_and_score_labels'] = 'PASS'
    checks['non_binary_dyadic_float64_roundtrip'] = 'PASS'
    return {'status':'PASS','checks':checks,'real_calibration_threshold_computed':False,'target_sets_computed':False,'seed':20260915}
