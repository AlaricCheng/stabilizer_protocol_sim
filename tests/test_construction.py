import pytest
import galois
from lib.construction import initialization, initialization_block, stabilizer_construction, qrc_construction
from lib.utils import rank, get_H_s, get_R_s
from lib.hypothesis import correlation_function

GF = galois.GF(2)


@pytest.mark.parametrize("n, m, g", [(20, 40, 2), (30, 70, 3), (40, 90, 4)])
def test_stabilizer_construction(n, m, g):
    H, s = initialization(n, m, g)
    assert rank(H) == n
    assert H.shape == (m, n) 

    H, s = stabilizer_construction(n, m, g, initAlg = 1)
    assert rank(H) == n
    assert H.shape == (m, n)

    R_s = get_R_s(H, s)
    H_s = get_H_s(H, s)

    assert (R_s @ s == 0).all()
    assert (H_s @ s == 1).all()

    assert rank(H_s.T @ H_s) == g
    assert abs(correlation_function(H, s)) == 2**(-g/2)

# @pytest.mark.skip
@pytest.mark.parametrize("n, m, g", [(20, 40, 2), (50, 70, 3)])
def test_stabilizer_construction_block(n, m, g):
    H, s = initialization_block(n, m, g, d1 = 0)
    assert rank(H) == n
    assert H.shape == (m, n) 

    H, s = stabilizer_construction(n, m, g, d1 = 0, initAlg = 2)
    assert rank(H) == n
    assert H.shape == (m, n)

    R_s = get_R_s(H, s)
    H_s = get_H_s(H, s)

    assert (R_s @ s == 0).all()
    assert (H_s @ s == 1).all()

    assert rank(H_s.T @ H_s) == g
    assert abs(correlation_function(H, s)) == 2**(-g/2)


def test_stabilizer_construction_block_specific():
    n, m, g, m1, d, d1 = 100, 200, 4, 100, 45, 45
    H, s = stabilizer_construction(n, m, g, m1, d, d1, initAlg = 2)
    assert rank(H) == n
    assert H.shape == (m, n)

    R_s = get_R_s(H, s)
    H_s = get_H_s(H, s)

    assert (R_s @ s == 0).all()
    assert (H_s @ s == 1).all()

    assert rank(H_s.T @ H_s) == g
    assert abs(correlation_function(H, s)) == 2**(-g/2)


def test_stabilizer_construction_block_specific_concat():
    n, m, g, m1, d, d1 = 100, 200, 4, 100, 45, 45
    H, s = stabilizer_construction(n, m, g, m1, d, d1, initAlg = 2, concat_D = True, m0 = 20, d0 = 9)
    assert rank(H) == n
    assert H.shape == (m, n)

    R_s = get_R_s(H, s)
    H_s = get_H_s(H, s)

    assert (R_s @ s == 0).all()
    assert (H_s @ s == 1).all()

    assert rank(H_s.T @ H_s) == g
    assert abs(correlation_function(H, s)) == 2**(-g/2)

    H, s = stabilizer_construction(n, m, g, m1, d, d1, initAlg = 2, concat_D = True, m0 = 20, d0 = 9, concat_C1 = True)
    assert rank(H) == n
    assert H.shape == (m, n)

    R_s = get_R_s(H, s)
    H_s = get_H_s(H, s)

    assert (R_s @ s == 0).all()
    assert (H_s @ s == 1).all()

    assert rank(H_s.T @ H_s) == g
    assert abs(correlation_function(H, s)) == 2**(-g/2)


@pytest.mark.parametrize("n, m, q", [(5, 15, 7), (15, 30, 23), (45, 60, 31), (75, 100, 47)])
def test_qrc_construction(n, m, q):
    H, s = qrc_construction(n, m, q)
    assert rank(H) == n

    H_s = get_H_s(H, s)
    R_s = get_R_s(H, s)
    assert (R_s @ s == 0).all()
    assert (H_s @ s == 1).all()
    assert rank(H_s.T @ H_s) == 1
    assert correlation_function(H, s) == 2**(-1/2)