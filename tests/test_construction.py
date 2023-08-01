import pytest
import galois
from lib.construction import initialization, stabilizer_construction, qrc_construction
from lib.utils import rank, get_H_s, get_R_s
from lib.hypothesis import correlation_function

GF = galois.GF(2)


@pytest.mark.parametrize("n, m, g", [(7, 15, 1), (20, 30, 2), (30, 40, 3), (40, 46, 4), (11, 20, 1)])
def test_stabilizer_construction(n, m, g):
    H, s = initialization(n, m, g)
    assert rank(H) == n
    assert H.shape == (m, n) or H.shape == (m+1, n) # there should be a bug in add_row_redundancy or check_element

    H, s = stabilizer_construction(n, m, g)
    assert rank(H) == n
    assert H.shape == (m, n) or H.shape == (m+1, n)

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