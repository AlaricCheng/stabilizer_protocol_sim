import pytest
import galois
from lib.gen_matrix import sample_D, sample_F
from lib.utils import hamming_weight, rank

GF = galois.GF(2)

@pytest.mark.parametrize("m1, d", [(30, 10), (40, 17), (50, 20)])
def test_sample_D(m1, d):
    D = sample_D(m1, d)
    assert D.shape == (m1, d) or D.shape == (m1, d-1)
    for c in D.T:
        assert hamming_weight(c) % 4 == 0
        assert (c.reshape(1, -1) @ D == 0).all()


@pytest.mark.parametrize("m1, g", [(10, 2), (33, 3), (50, 4)])
def test_sample_F(m1, g):
    d = int((m1 - g)/2) - 2
    D = sample_D(m1, d)
    F = sample_F(m1, g, D)

    assert F.shape == (m1, g)
    assert (D.T @ F == 0).all()
    assert rank(F.T @ F) == g
