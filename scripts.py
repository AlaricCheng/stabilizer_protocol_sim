import numpy as np
import galois
import time
import argparse
import json

from lib.construction import qrc_construction, stabilizer_construction
from lib.attack import extract_one_secret, sampling_with_H
from lib.hypothesis import hypothesis_test, bias
from lib.utils import get_H_s, rank
from lib.parallel import *

GF = galois.GF(2)

def fix_SB(q, max_iter = 2**15, n_repeats = 100):
    r = int((q+1)/2)
    m = 2*q
    g = 1
    
    res = {}
    for n in range(r, r+q, 2):
        print("n =", n)
        res[n] = []
        def helper(_):
            H, s = qrc_construction(n, m, q)
            beta = bias(H, s)
            s_candidate, count = extract_one_secret(H, max_iter = max_iter, check = "qrc")
            if s_candidate is None:
                return False, count
            else:
                X = sampling_with_H(H, s_candidate, 5000)
                return hypothesis_test(s, X, beta), count
        
        res[n] = tpmap(Task(helper), range(n_repeats), desc = "n = {}".format(n))

    return res


def compared_with_qrc(q, max_iter = 2**15, n_repeats = 100):
    r = int((q+1)/2)
    m = 2*q
    g = 1

    res = {}
    for n in range(r, r+q, 2):
        print("n =", n)
        res[n] = []
        def helper(_):
            H, s = stabilizer_construction(n, m, g)
            beta = bias(H, s)
            s_candidate, count = extract_one_secret(H, max_iter = max_iter, check = "rank")
            if s_candidate is None:
                return False, count
            else:
                X = sampling_with_H(H, s_candidate, 5000)
                return hypothesis_test(s, X, beta), count
        
        res[n] = tpmap(Task(helper), range(n_repeats), desc = "n = {}".format(n))

    return res


def qrc_kernel(q, n_repeats = 100):
    m = 2*q
    n = int((q+1)/2) + q

    def helper(_):
        H, s = qrc_construction(n, m, q)
        d = GF.Random(n)
        H_d = get_H_s(H, d)
        G_d = H_d.T @ H_d

        return n - rank(G_d)
    
    return tpmap(Task(helper), range(n_repeats))


def stab_kernel(n, m, g, n_repeats = 100):

    def helper(_):
        H, s = stabilizer_construction(n, m, g)
        print(rank(H))
        d = GF.Random(n)
        H_d = get_H_s(H, d)
        G_d = H_d.T @ H_d

        return n - rank(G_d)
    
    return tpmap(Task(helper), range(n_repeats))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_SB", action = "store_true")
    parser.add_argument("--compared_with_qrc", action = "store_true")
    parser.add_argument("--qrc_kernel", action = "store_true")
    parser.add_argument("--stab_kernel", action = "store_true")
    parser.add_argument("--q", type = int, default = 7)
    parser.add_argument("--g", type = int, default = 1)
    parser.add_argument("--n", type = int)
    parser.add_argument("--m", type = int)

    args = parser.parse_args()
    print(args)

    if args.fix_SB:
        res = fix_SB(args.q)
        with open(f"./data/fix_SB_{args.q}.json", "w") as f:
            json.dump(res, f)
    if args.compared_with_qrc:
        res = compared_with_qrc(args.q)
        with open(f"./data/compared_with_qrc_{args.q}.json", "w") as f:
            json.dump(res, f)
    if args.qrc_kernel:
        res = qrc_kernel(args.q)
        with open(f"./data/qrc_kernel_{args.q}.json", "w") as f:
            json.dump(res, f)
    if args.stab_kernel:
        if args.n is None:
            n = int((args.q+1)/2) + args.q
        if args.m is None:
            m = n + 50
        
        res = stab_kernel(n, m, args.g)
        with open(f"./data/stab_kernel_{n}_{m}_{args.g}.json", "w") as f:
            json.dump(res, f)
