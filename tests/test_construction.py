import pytest
import numpy as np
from numpy.random import default_rng
import galois
from lib.construction import *

GF = galois.GF(2)

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


def test_add_row_redundancy():
    s = GF([1,0,1,0,1])
    H_M = random_main_part(5, 2, s)
    m = len(H_M)
    H = add_row_redundancy(H_M, s, 10)
    a = GF.Zeros((len(H), 1))
    for i in range(m):
        a[i] = 1
    assert np.all(H @ s.reshape(-1, 1) == a)


def test_add_col_redundancy():
    s = GF([1,0,1,0,1])
    H_M = random_main_part(5, 4, s)
    H_M, s = add_col_redundancy(H_M, s, 10)
    assert H_M.shape == (4, 15)
    assert np.all(H_M @ s.reshape(-1, 1) == 1)


@pytest.mark.parametrize("n", [10, 20, 30])
@pytest.mark.parametrize("g", [1, 2, 3])
class TestFactorization:
    def test_final_factor(self, n, g):
        s = GF.Random(n)
        tab = random_tableau(n, g, s)
        fac = Factorization(tab.copy(), s)
        H = fac.final_factor(rand_rows=10)
        assert np.all(tab[:, :n] == H.T @ H)
        assert np.all(H @ s.reshape(-1, 1) == 1)

        fac1 = Factorization(tab.copy(), s)
        for row in H:
            fac1.backward_evolution(row)
        assert np.all(fac1.G == 0)
        assert np.all(fac1.tab[:, 2*n] == 0)


def test_generate_QRC_instance():
    H, s = generate_QRC_instance(7, 7, 5)
    H_M = H[(H @ s == 1)]
    assert rank(H_M.T @ H_M) == 1


def test_generate_stab_instance():
    H, s = generate_stab_instance(15, 3, exp_nullity = 5)
    assert H.shape[1] - H.shape[0]/2 <= 5
    assert H.shape[1] == 15
    H_M = H[(H @ s == 1)]
    assert rank(H_M.T @ H_M) <= 3
    