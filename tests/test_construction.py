import pytest
from lib.construction import *

@pytest.mark.parametrize("n", [10, 20, 30])
@pytest.mark.parametrize("g", [1, 2, 3])
def test_random_tableau(n, g):
    rank = lambda A: np.sum(~np.all(A.row_reduce() == 0))
    s = GF.Random(n)
    tab = random_tableau(n, g, s)
    G = tab[:, :n]
    r = tab[:, 2*n:]
    assert rank(G) == rank(np.hstack((G, r)))