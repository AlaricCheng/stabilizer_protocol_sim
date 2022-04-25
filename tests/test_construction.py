import pytest
from lib.construction import *

rank = lambda A: np.sum(~np.all(A.row_reduce() == 0))

@pytest.mark.parametrize("n", [10, 20, 30])
@pytest.mark.parametrize("g", [1, 2, 3])
def test_random_object(n, g):
    '''
    Test random_main_part, random_gram, and random_tableau.
    For the tableau, test whether the generated stabilizer tableau is `consistent', i.e., overlap is not zero.
    '''
    s = GF.Random(n)
    H = random_main_part(n, g, s, seed = default_rng(0))
    assert rank(H) <= g # test rank of the main part
    assert np.all(H @ s.reshape(-1, 1) == 1)

    G = random_gram(n, g, s, seed = default_rng(0))
    assert rank(G) == rank(H) # test rank of the Gram matrix

    tab = random_tableau(n, g, s, seed = default_rng(0))
    assert np.all(G == tab[:, :n]) # test the X-part of the stabilizer tableau
    r = tab[:, 2*n:]
    assert rank(G) == rank(np.hstack((G, r))) # test consistency 


def test_add_redundancy():
    pass


@pytest.mark.parametrize("n", [10, 20, 30])
@pytest.mark.parametrize("g", [1, 2, 3])
class TestFactorization:
    def test_final_factor(self, n, g):
        s = GF.Random(n)
        tab = random_tableau(n, g, s)
        fac = Factorization(tab, s)
        H = fac.final_factor()
        assert np.all(tab[:, :n] == H.T @ H)
        assert np.all(H @ s.reshape(-1, 1) == 1)