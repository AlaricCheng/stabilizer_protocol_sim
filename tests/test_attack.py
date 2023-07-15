import pytest
import galois
import lib
from lib.mat_factor import generate_QRC_instance
from lib.attack import QRCAttack, LinearityAttack


GF = galois.GF(2)

class TestLinearityAttack:
    @pytest.mark.parametrize("q", [7, 23, 31])
    def test_classical_sampling(self, q):
        H, s = generate_QRC_instance(q, q, 1)
        X = LinearityAttack(H).classical_sampling(1000, budget = 500, g_thres = 1)
        assert lib.hypothesis_test(s, X, 0.854)


class TestQRCAttack:
    @pytest.mark.parametrize("q", [7, 23, 31])
    def test_classical_sampling(self, q):
        H, s = generate_QRC_instance(q, q, 1)
        # S = GF.Random((2, int((q+3)/2)))
        # S = np.vstack((s.reshape(1, -1), S))
        X = QRCAttack(H).classical_sampling(1000, one_sol = True)
        assert lib.hypothesis_test(s, X, 0.854)

        X = QRCAttack(H).classical_sampling(1000, budget = 500)
        assert lib.hypothesis_test(s, X, 0.854)


def test_classical_samp_same_bias():
    s = GF.Random(10, seed = 0)
    X = lib.attack.classical_samp_same_bias(s, 0.7, 1000)
    assert lib.hypothesis_test(s, X, 0.7)


def test_classical_samp_diff_bias():
    S = GF.Random((3, 10), seed = 0)
    beta = [0.7, 0.8, 0.9]
    X = lib.attack.classical_samp_diff_bias(S, beta, 1000)
    for s, b in zip(S, beta):
        assert lib.hypothesis_test(s, X, b)

    beta = [0.7, 0.7, 0.8]
    X = lib.attack.classical_samp_diff_bias(S, beta, 1000)
    for s, b in zip(S, beta):
        assert lib.hypothesis_test(s, X, b)

    beta = [0.7]
    X = lib.attack.classical_samp_diff_bias(S[0], beta, 1000)
    assert lib.hypothesis_test(S[0], X, beta[0])