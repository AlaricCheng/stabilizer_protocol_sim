import os
import galois
import argparse
import json

from lib.construction import qrc_construction, stabilizer_construction
from lib.attack import extract_one_secret, sampling_with_H
from lib.hypothesis import hypothesis_test, bias
from lib.utils import get_H_s, rank
from lib.parallel import *
from time import time

GF = galois.GF(2)

def fix_SB(q, max_iter = 2**15, n_repeats = 100):
    r = int((q+1)/2)
    m = 2*q
    
    res = {}
    for n in range(r, r+q, 2):
        print("n =", n)
        res[n] = []
        def helper(_):
            H, s = qrc_construction(n, m, q)
            s_candidate, count = extract_one_secret(H, max_iter = max_iter, check = "qrc")
            if s_candidate is None:
                return False, count
            else:
                X = sampling_with_H(H, s_candidate, 5000)
                beta = 0.8535
                return hypothesis_test(s, X, beta), count
        
        res[n] = tpmap(Task(helper), range(n_repeats), desc = "n = {}".format(n))

    return res


def compared_with_qrc(q, max_iter = 2**15, n_repeats = 100):
    r = int((q+1)/2)
    m = q + r + 50
    g = 3

    res = {}
    for n in range(r, r+q, 5):
        print("n =", n)
        res[n] = []
        def helper(_):
            H, s = stabilizer_construction(n, m, g)
            s_candidate, count = extract_one_secret(H, max_iter = max_iter, check = "rank", g_thres = g)
            if s_candidate is None:
                return False, count
            else:
                beta = bias(H, s)
                X = sampling_with_H(H, s_candidate, 5000)
                return hypothesis_test(s, X, beta), count
        
        res[n] = tpmap(Task(helper), range(n_repeats), desc = "n = {}".format(n))

    return res


def stab_scheme(g, g_thres, n_init = 50, m = 200, max_iter = 2**15, n_repeats = 100):
    res = {}

    for n in range(n_init, m - 30, 5):
        print("n =", n)
        res[n] = []
        def helper(_):
            H, s = stabilizer_construction(n, m, g)
            s_candidate, count = extract_one_secret(H, max_iter = max_iter, check = "rank", g_thres = g_thres)
            if s_candidate is None:
                return False, count
            else:
                beta = bias(H, s)
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
        d = GF.Random(n)
        H_d = get_H_s(H, d)
        G_d = H_d.T @ H_d
        print(rank(H), n - rank(G_d))

        return n - rank(G_d)
    
    return tpmap(Task(helper), range(n_repeats))


def stab_scheme_kernel(g, n_init = 100, m = 200, n_repeats = 100):
    res = {}

    for n in range(n_init, m - 30, 5):
        print("n =", n)
        res[n] = stab_kernel(n, m, g, n_repeats = n_repeats)

    return res

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--n-proc", type=int, default=-1, help="number of processes to run in parallel. Default is -1, meaning using all available CPUs.")
    parser.add_argument("--fix_SB", action = "store_true")
    parser.add_argument("--compared_with_qrc", action = "store_true")
    parser.add_argument("--stab_scheme", action = "store_true")
    parser.add_argument("--qrc_kernel", action = "store_true")
    parser.add_argument("--stab_kernel", action = "store_true")
    parser.add_argument("--stab_scheme_kernel", action = "store_true")
    parser.add_argument("--q", type = int, default = 7)
    parser.add_argument("--g", type = int, default = 1)
    parser.add_argument("--g_thres", type = int, default = 1)
    parser.add_argument("--n", type = int)
    parser.add_argument("--m", type = int)

    
    args = parser.parse_args()
    print(args)
    
    n_proc = os.cpu_count() if args.n_proc == -1 else args.n_proc
    global_settings(processes = n_proc) # set the number of processes to run in parallel
    
    tick = time()
    if args.fix_SB:
        res = fix_SB(args.q)
        with open(f"./data/fix_SB_{args.q}.json", "w") as f:
            json.dump(res, f)
    if args.compared_with_qrc:
        res = compared_with_qrc(args.q)
        with open(f"./data/compared_with_qrc_{args.q}.json", "w") as f:
            json.dump(res, f)
    if args.stab_scheme:
        res = stab_scheme(args.g, args.g_thres)
        with open(f"./data/stab_scheme_{args.g}_{args.g_thres}.json", "w") as f:
            json.dump(res, f)
    if args.stab_scheme_kernel:
        res_kernel = stab_scheme_kernel(args.g)
        with open(f"./data/stab_scheme_kernel_{args.g}.json", "w") as f:
            json.dump(res_kernel, f)
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

    print(f"Time elapsed: {(time() - tick)/3600:.2f} h")
