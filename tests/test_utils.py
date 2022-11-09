# import pytest
import numpy as np
from numpy.random import default_rng
import lib
from lib.utils import wrap_seed, solvesystem
import galois

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

def test_KF_partition():
    A = GF.Random((6, 3))
    print(A)
    K, F = lib.utils.KF_partition(A)
    assert np.all(GF.Ones((1, len(K))) @ K == 0)

def test_lempel_sequence():
    A = GF.Random((6, 3))
    print(A)
    G = A.T @ A
    seq = lib.utils.lempel_sequence(A)
    for E in seq:
        assert np.all(E.T @ E == G)

