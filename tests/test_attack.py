import pytest
import galois
import lib
from lib.construction import qrc_construction, stabilizer_construction
from lib.utils import get_H_s
from lib.attack import classical_samp_same_bias, classical_samp_diff_bias, qrc_check, property_check, extract_one_secret, naive_sampling, sampling_with_H
from lib.hypothesis import bias

GF = galois.GF(2)

@pytest.mark.parametrize("q", [7, 23, 31, 47])
def test_qrc_check(q):
    n = int((q+1)/2)
    m = 2*q
    H, s = qrc_construction(n, m, q)

    assert qrc_check(H, s)

    H = GF.Random((m, n), seed = 0)
    assert not qrc_check(H, s)


@pytest.mark.parametrize("q", [7, 23, 31, 47])
def test_property_check(q):
    n = int((q+1)/2)
    m = 2*q
    H, s = qrc_construction(n, m, q)

    assert property_check(H, s, 1)

    H, s = stabilizer_construction(n, m, 2)

    assert property_check(H, s, 2)


@pytest.mark.parametrize("q", [23, 31, 47])
def test_extract_one_secret(q):
    n = int((q+1)/2)
    m = 2*q
    H, s = qrc_construction(n, m, q)

    s_candidate, count = extract_one_secret(H, g_thres = 1, max_iter = 2**10, check = "qrc")

    assert (s == s_candidate).all()


def test_sampling():
    n = 20
    m = 30
    H, s = stabilizer_construction(n, m, 1)

    beta = bias(H, s)
    
    X = naive_sampling(H, s, 10000)
    assert lib.hypothesis_test(s, X, beta)

    X = sampling_with_H(H, s, 10000)
    assert lib.hypothesis_test(s, X, beta)


def test_classical_samp_same_bias():
    s = GF.Random(10, seed = 0)
    X = classical_samp_same_bias(s, 0.7, 1000)
    assert lib.hypothesis_test(s, X, 0.7)


def test_classical_samp_diff_bias():
    S = GF.Random((3, 10), seed = 0)
    beta = [0.7, 0.8, 0.9]
    X = classical_samp_diff_bias(S, beta, 1000)
    for s, b in zip(S, beta):
        assert lib.hypothesis_test(s, X, b)

    beta = [0.7, 0.7, 0.8]
    X = classical_samp_diff_bias(S, beta, 1000)
    for s, b in zip(S, beta):
        assert lib.hypothesis_test(s, X, b)

    beta = [0.7]
    X = classical_samp_diff_bias(S[0], beta, 1000)
    assert lib.hypothesis_test(S[0], X, beta[0])