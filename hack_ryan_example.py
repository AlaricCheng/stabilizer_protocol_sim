import galois
import lib
from lib.attack import QRCAttack
from lib.utils import load_data

GF = galois.GF(2)

if __name__ == "__main__":
    H, s = load_data("./examples/ryan_7.3.xprog")
    print("Real secret:", s)
    X = QRCAttack(H).classical_sampling_enhanced(1000, budget = 500, verbose = True)
    print("Pass the test?", lib.hypothesis_test(s, X, 0.854))