# %%
import argparse
import numpy as np
from numpy.random import default_rng
import galois
import lib
from lib.utils import solvesystem, rank
from lib.construction import generate_QRC_instance
import matplotlib.pyplot as plt
from time import perf_counter
from tqdm import tqdm, trange
from lib.parallel import *

GF = galois.GF(2)

def attack(H, s, num_samples = 1000, correct_d = False):
    KMA = lib.QRCAttack(H)
    # whether to set correct d manually
    while KMA.check_d(s) == False and correct_d == True:
        KMA.regenerate_d()
    print("Correct d: ", KMA.check_d(s))
    d = KMA.get_d()
    H_M = KMA.get_P_s(s)
    # print useful information
    print("Rank of H_d: ", rank(H_M[(H_M @ d).nonzero()[0]]))
    print("Rank of the Gram matrix: ", KMA.get_Gs_rank(s))
    # begin attack
    print("====== Attack ======")
    M = KMA.get_M(num_samples)
    S = solvesystem(M)
    print("Dimension of solution space: ", len(S))
    if len(S) < 5:
        S = solvesystem(M, all_sol=True)
    for s0 in S[:10]:
        _rank = KMA.get_Gs_rank(s0)
        if _rank == KMA.get_Gs_rank(s) and np.all(s0 == s):
            print("secret found: s = ", s0)
            break
        print(_rank)


def generate_stab_instance(
    n: int, 
    g: int, 
    rd_row: int = 0, 
    rd_col: int = 0,
    obf_rows: int = 10
):
    '''
    Args:
        n: number of columns in the main part before adding column redundancy
        g: rank of the Gram matrix before adding column redundancy
        rd_row: number of redundant rows
        rd_col: number of redundant columns
        obf_rows: number of random rows for obfuscating the tableau
    '''
    s = GF.Random(n)
    if np.all(s == 0): # ensure s will not be all zero
        s[0] = 1
    seed = default_rng().choice(100)
    H_M = lib.random_main_part(n, g, s, seed = seed)
    G = lib.random_gram(n, g, s, seed = seed)
    tab = lib.random_tableau(n, g, s, seed = seed)

    sc = lib.Factorization(tab, s)
    H_M = sc.final_factor(obf_rows=obf_rows)
    assert np.all(H_M.T @ H_M == G)
    print()
    assert np.all(H_M @ s.reshape(-1, 1) == 1) # check codeword
    H_M, s = lib.add_col_redundancy(H_M, s, rd_col)
    print("rank of H_M:", rank(H_M), "\tshape H_M:", H_M.shape)
    H = lib.add_row_redundancy(H_M, s, rd_row)
    print("rank of H:", rank(H), "\tshape of H:", H.shape)

    return H, s


def sol_size_qrc(q):
    '''
    Get size of solution set
    '''
    dim_sol = []
    for rd_col in trange(1, q, 10):
        tmp = []
        H, s = generate_QRC_instance(q, q, rd_col)
        for _ in range(10):
            KMA = lib.QRCAttack(H)
            M = KMA.get_M(1000)
            S = solvesystem(M)
            tmp.append(len(S))
        dim_sol.append(tmp)
    
    return dim_sol


def attack_succ_qrc_enhanced(q, thres = 15, rep = 100, lim = 40):
    '''
    Probability that the real secret is in the candidate set
    Args
        lim: 
    '''
    start_pt, end_pt = int((q+1)/2 - lim/2), int((q+1)/2 + lim)
    succ_prob = {}
    for rd_col in range(start_pt, end_pt, 4):
        H, s = generate_QRC_instance(q, q, rd_col)
        # print("Secret:", s)
        count = 0
        for _ in trange(rep, desc = f"q = {q}, {rd_col} redundant columns"):
            X = lib.QRCAttack(H).extract_secret_enhanced(10*H.shape[0], budget=2**thres)
            if lib.hypothesis_test(s, X, 0.854):
                count += 1
        succ_prob[rd_col] = count/rep

    return succ_prob


def attack_succ_qrc_test(q, rep = 100, lim = 40) -> dict:
    '''
    Probability that the prover's samples pass the verifier's test
    Args
        lim: 
    '''
    start_pt, end_pt = int((q+1)/2 - lim/2), int((q+1)/2 + lim)
    succ_prob = {}
    for rd_col in trange(start_pt, end_pt, 4):
        H, s = generate_QRC_instance(q, q, rd_col)
        # print("Secret:", s)
        count = 0
        bad_d = 0
        for _ in range(rep):
            KMA = lib.QRCAttack(H)
            while KMA.check_d(s) == False:
                KMA.regenerate_d()
            M = KMA.get_M(1000)
            if M is None:
                bad_d += 1
                continue
            if rank(M) == M.shape[1]:
                bad_d += 1
                continue
            S = solvesystem(M)
            cand_s = GF.Random((1, len(S))) @ S
            X = KMA.classical_sampling(cand_s, 5000)
            if lib.hypothesis_test(s, X, 0.854):
                count += 1
        succ_prob[rd_col] = count/(rep - bad_d)

    return succ_prob


def attack_succ_qrc_no_redundant(q, check = True):
    '''
    Probability that the prover's samples pass the verifier's test (original QRC construction)
    Args
        check (bool): whether to check weights of the resulting code
    '''
    H, s = generate_QRC_instance(q, q, 0)
    count = 0
    bad_d = 0
    for _ in trange(args.rep):
        KMA = lib.QRCAttack(H)
        while KMA.check_d(s) == False:
            KMA.regenerate_d()
        M = KMA.get_M(1000)
        if M is None:
            bad_d += 1
            continue
        if rank(M) == M.shape[1]:
            bad_d += 1
            continue
        if check:
            S = solvesystem(M, all_sol = True)
            cand_s = KMA.print_candidate_secret(S)
        else:
            S = solvesystem(M)
            cand_s = GF.Random((1, len(S))) @ S
        X = KMA.classical_sampling(cand_s, 5000)
        if lib.hypothesis_test(s, X, 0.854):
            count += 1
    return count/(args.rep - bad_d)


def merge_dict(dicts: list[dict]) -> list:
    '''
    dicts contains dictionaries with the same keys. Merge the values into a list.
    '''
    merged_dict = {}
    for key in dicts[0]:
        merged_dict[key] = [ele[key] for ele in dicts]
    
    merged_list = [[key] + value for key, value in zip(merged_dict.keys(), merged_dict.values())]

    return np.array(merged_list)


def draw_fig(idx):
    if idx == 1:
        fig, ax = plt.subplots()
        for q in [103, 127, 151, 167, 191]:
            dim_sol = sol_size_qrc(q) 
            ax.errorbar(range(1, q, 10), np.mean(dim_sol, axis = 1), yerr=np.std(dim_sol, axis = 1), marker = "*", label = f"q = {q}")

        ax.set_xlabel("Number of redundant columns")
        ax.set_ylabel("Dimension of solution space")
        ax.legend()

        fig.savefig("./fig/sol_size.svg", bbox_inches = "tight")
    elif idx == 2:
        def helper(q):
            return attack_succ_qrc_enhanced(q, thres = args.thres, rep = args.rep)

        fig, ax = plt.subplots()
        succ_prob_all = []
        for q in [103, 127, 151, 167, 191]:
            succ_prob_tmp = tpmap(Task(helper), [q]*20, desc = f"q = {q}")
            succ_prob = merge_dict(succ_prob_tmp)
            ax.errorbar(succ_prob[:, 0], np.mean(succ_prob[:, 1:], axis = 1), yerr=np.std(succ_prob[:, 1:], axis = 1), marker = "^", label = f"q = {q}")
            succ_prob_all.append(succ_prob)

        np.save("./succ_prob_enhanced.npy", succ_prob_all)

        ax.set_xlabel("Number of redundant columns")
        ax.set_ylabel("Success probability")
        ax.legend()

        fig.savefig("./fig/succ_prob_enhanced.svg", bbox_inches = "tight")
    elif idx == 3:
        def helper(q):
            return attack_succ_qrc_test(q, rep = args.rep)

        fig, ax = plt.subplots()
        succ_prob_all = []
        for q in [103, 127, 151, 167, 191]:
            succ_prob_tmp = tpmap(Task(helper), [q]*20, desc = f"q = {q}")
            succ_prob = merge_dict(succ_prob_tmp)
            ax.errorbar(succ_prob[:, 0], np.mean(succ_prob[:, 1:], axis = 1), yerr=np.std(succ_prob[:, 1:], axis = 1), marker = "^", label = f"q = {q}")

            ref_pt = np.mean([attack_succ_qrc_no_redundant(q, False) for _ in range(20)])
            ax.plot(succ_prob[:, 0], [ref_pt]*len(succ_prob), linestyle = "--", color = "lightcoral")
            succ_prob_all.append(succ_prob)

        np.save("./succ_prob_test.npy", succ_prob_all)

        ax.set_xlabel("Number of redundant columns")
        ax.set_ylabel("Success probability")
        ax.legend()

        fig.savefig("./fig/succ_prob_test.svg", bbox_inches = "tight")




# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--n-proc", type = int, default = 3, help = "number of processors used")
    group = parser.add_mutually_exclusive_group()

    # QRC
    group.add_argument("-Q", "--QRC", action = "store_true", help = "for QRC construction")
    group_qrc = parser.add_argument_group("QRC")
    group_qrc.add_argument("-q", type = int, help = "size parameter for QRC")

    # stabilizer
    group.add_argument("-s", "--stabilizer", action = "store_true", help = "for stabilizer construction")
    group_stab = parser.add_argument_group("stabilizer construction")
    group_stab.add_argument("-n", type = int, help = "number of qubits in the initial main part")
    group_stab.add_argument("-g", type = int, help = "rank of the Gram matrix")
    group_stab.add_argument("--obf-rows", type = int, default = 10, help = "number of random rows for obfuscating the tableau")

    # redundancy
    parser.add_argument("--rd-row", type = int, help = "number of redundant rows")
    parser.add_argument("--rd-col", type = int, help = "number of redundant columns")

    # attack
    group_attack = parser.add_argument_group("attack")
    group_attack.add_argument("-a", "--attack", action = "store_true", help = "attack")
    group_attack.add_argument("--correct-d", action = "store_true", help = "whether to set correct d manually")

    parser.add_argument("-v", "--verbose", action = "store_true", help = "verbose")

    # figure
    group_fig = parser.add_argument_group()
    group_fig.add_argument("--fig", type = int, help = "draw figure")
    group_fig.add_argument("--thres", type = int, default = 15, help = "Threshold of solution space dimension")
    group_fig.add_argument("--rep", type = int, default=100, help = "Number of repetitions for attacking each instance")

    parser.add_argument("--test", action = "store_true", help = "for test")

    args = parser.parse_args()

    global_settings(processes = args.n_proc)
    
    # generate H and s
    if args.QRC == True:
        H, s = generate_QRC_instance(args.q, rd_row = args.rd_row, rd_col = args.rd_col)
    elif args.stabilizer == True:
        H, s = generate_stab_instance(args.n, args.g, rd_row = args.rd_row, rd_col = args.rd_col, obf_rows = args.obf_rows)

    if args.verbose:
        print("Matrix:\n", H)
        print("Secret:\n", s)

    # attack
    if args.attack:
        attack(H, s, correct_d = args.correct_d)

    # figure
    if args.fig:
        draw_fig(args.fig)

    if args.test:
        def f_test(q):
            return attack_succ_qrc_no_redundant(q, True)
        # a = tpmap(f_test, [7]*10)
        # a = attack_succ_qrc_no_redundant(31, False)
        a = {1:0, 2: 2}
        b = {1:2, 2: 3}
        c = {1:1, 2: 4}
        print(merge_dict([a,b,c]))
        # print(list(zip([7]*10, [False]*10)))
