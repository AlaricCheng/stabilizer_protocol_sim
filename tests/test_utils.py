import numpy as np
import galois
from numpy.random import default_rng
from lib.utils import wrap_seed, solvesystem, random_solution, get_D_space, get_H_s, check_D_doubly_even
from lib.gen_matrix import sample_D
from lib.construction import qrc_construction, stabilizer_construction


GF = galois.GF(2)


def test_wrap_seed():
    rng1 = default_rng(0)
    rng2 = wrap_seed(default_rng(0))
    a = rng1.choice(10, size = 10)
    b = rng2.choice(10, size = 10)
    assert np.all(a == b)

def test_solvesystem():
    A = GF.Random((4, 5), seed = 0)
    sol = solvesystem(A)
    assert len(sol) == 1
    assert np.all(sol[0] == GF([0, 1, 0, 1, 0]))

    b = GF([1,0,1,0])
    sol = solvesystem(A, b, all_sol=True)
    for s in sol:
        assert np.all(A @ s.reshape(-1, 1) == b.reshape(-1, 1))

    for _ in range(10):
        s = random_solution(A, b)
        assert np.all(A @ s.reshape(-1, 1) == b.reshape(-1, 1))


def test_get_D_space():
    H, s = qrc_construction(5, 14, 7)
    H_s = get_H_s(H, s)
    D = get_D_space(H_s, 1)

    assert check_D_doubly_even(D)

    H, s = stabilizer_construction(7, 14, 2)
    H_s = get_H_s(H, s)
    D = get_D_space(H_s, 2)

    assert check_D_doubly_even(D)


def test_check_D_doubly_even():
    D = sample_D(8, 3)
    assert check_D_doubly_even(D)
