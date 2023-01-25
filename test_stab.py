import galois
import numpy as np
import lib
from lib.utils import rank, bias
from time import time
import argparse
from tqdm import trange

GF = galois.GF(2)

def stab_test():
    for _ in trange(3):
        H, s = lib.generate_stab_instance_no_red_col(150, 2, verbose=True)
        print(s)
        LA = lib.LinearityAttack(H)
        X, c = LA.classical_sampling(5000, budget=2**18, g_thres=5, require_count=True, verbose=True)
        print(c, "Pass or not:", lib.hypothesis_test(s, X, bias(LA.get_Gs_rank(s))))
        # S, c = LA.extract_secret(10*LA.P.shape[0], g_thres=5, verbose = True)
        # print(len(S), c)
        # print("secret in candidate set?", np.array([(a == s).all() for a in S]).any())

def qrc_test():
    for _ in trange(3):
        H, s = lib.generate_QRC_instance(71, 71, 50, verbose=True)
        print("True secret:", s)
        QRCA = lib.QRCAttack(H)
        X, c = QRCA.classical_sampling_original(5000, budget=2**15, require_count=True, verbose=True)
        print(c, "Pass or not:", lib.hypothesis_test(s, X, 0.854))
        S, c = QRCA.extract_secret(10*QRCA.P.shape[0])
        print(len(S), c)
        print(S)
        print("same QRC?", [(H @ s.reshape(-1, 1) == H @ a.reshape(-1, 1)).all() for a in S])
        print("secret in candidate set?", np.array([(a == s).all() for a in S]).any())
        print("\n====================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stab", action="store_true", help="run stab test")
    parser.add_argument("--qrc", action="store_true", help="run qrc test")

    args = parser.parse_args()

    if args.stab:
        stab_test()
    if args.qrc:
        qrc_test()
    