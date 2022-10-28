# %%
import argparse
import numpy as np
from numpy.random import default_rng
import galois
import lib
from lib.utils import solvesystem, rank
import matplotlib.pyplot as plt
from time import perf_counter
from tqdm import tqdm, trange
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

GF = galois.GF(2)

def attack(H, s, num_samples = 1000, correct_d = False):
    KMA = lib.KMAttack(H)
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


def generate_QRC_instance(
    q: int, 
    rd_row: int = 0, 
    rd_col: int = 0
):
    '''
    Args:
        q: size parameter for QRC
        rd_row: number of redundant rows
        rd_col: number of redundant columns
    '''
    QRC = lib.QRCConstruction(q) # initialization
    QRC.obfuscation(1000)
    H_M = QRC.P_s
    s = QRC.s
    H_M, s = lib.add_col_redundancy(H_M, s, rd_col)
    H = lib.add_row_redundancy(H_M, s, rd_row)
    print("rank of H_M:", rank(H_M), "\trank of H:", rank(H), "\tshape of H:", H.shape)

    return H, s


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
        with HiddenPrints():
            H, s = generate_QRC_instance(q, q, rd_col)
        for _ in range(10):
            KMA = lib.KMAttack(H)
            M = KMA.get_M(1000)
            S = solvesystem(M)
            tmp.append(len(S))
        dim_sol.append(tmp)
    
    return dim_sol


def attack_succ_qrc(q, thres = 15, rep = 20, lim = 30):
    '''
    Args
        lim: 
    '''
    start_pt = int((q+1)/2) - 15
    succ_prob = {}
    for rd_col in trange(start_pt, start_pt+lim, 3):
        with HiddenPrints():
            H, s = generate_QRC_instance(q, q, rd_col)
        # print("Secret:", s)
        count = 0
        for _ in range(rep):
            KMA = lib.KMAttack(H)
            M = KMA.get_M(1000)
            # print(len(M))
            S = solvesystem(M)
            # print("Correct d:", KMA.check_d(s), "\tDimension of solution space:", len(S))
            if len(S) < thres:
                S = solvesystem(M, all_sol=True)
            cand_s = KMA.print_candidate_secret(S)
            # print(cand_s)
            if s.tolist() in cand_s.tolist():
                count += 1
        succ_prob[rd_col] = count/rep

    return succ_prob


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
        fig, ax = plt.subplots()
        for q in [103, 127, 151, 167, 191]:
            succ_prob = attack_succ_qrc(q, thres = args.thres, rep = args.rep)
            ax.plot(succ_prob.keys(), succ_prob.values(), "^", label = f"q = {q}")

        ax.set_xlabel("Number of redundant columns")
        ax.set_ylabel("Success probability")
        ax.legend()

        fig.savefig("./fig/succ_prob.svg", bbox_inches = "tight")




# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    group_fig.add_argument("--rep", type = int, default=20, help = "Number of repetitions for attacking each instance")

    args = parser.parse_args()
    
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
    



exit()

# %%
if __name__ == "__main__1":
    P = np.loadtxt("./examples/4qubit_1110.prog", dtype = int)
    # P = np.loadtxt("../examples/5qubit_QRC.prog", dtype = int)
    # P = np.loadtxt("../examples/challenge.xprog", dtype = int)
    # print(GF(P))
    KMA = KMAttack(P)
    # KMA.regenerate_d()
    M = KMA.get_M(1000)
    S = solvesystem(M, all_sol=True)
    print(KMA.print_candidate_secret(S, print_rank=True))


# %%
if __name__ == "__main__":
    '''
    Test stabilizer construction
    '''
    # tab = np.loadtxt("./examples/4qubit_rank_1.tab", dtype = int)
    n = 100
    seed = default_rng(1).integers(1000)
    s = GF.Random(n, seed=seed)
    g = 5
    seed = default_rng(seed).integers(1000)
    H_M = random_main_part(n, g, s, seed = seed)
    G = random_gram(n, g, s, seed = seed)
    tab = random_tableau(n, g, s, seed = seed)
    
    sc = Factorization(tab, s)
    H_M = sc.final_factor()
    # H_M = sc.injecting_ones(H_M, s)
    assert np.all(H_M @ s.reshape(-1, 1) == 1) # check codeword
    assert np.all(H_M.T @ H_M == G)

    seed = default_rng(seed).integers(1000)
    H = add_redundancy(H_M, s, 5*np.ceil(np.log(n)), seed=seed)
    print("Rows of H: ", len(H))
    # H = H_M
    # print(H[5])

    # KMA = KMAttack(H)
    # while KMA.check_d(s) == False:
    #     KMA.regenerate_d()
    # print("Correct d: ", KMA.check_d(s))
    # print("Rank of the Gram matrix: ", KMA.get_Gs_rank(s))
    # M = KMA.get_M(1000)
    # S = solvesystem(M)
    # print("Dimension of solution space: ", len(S))
    # if len(S) < 10:
    #     S = solvesystem(M, all_sol=True)
    # for s0 in S[:10]:
    #     _rank = KMA.get_Gs_rank(s0)
    #     if _rank == KMA.get_Gs_rank(s):
    #         print(np.all(s0 == s))
    #     print(_rank)

    # try another attack
    print("Rank of H_M:", rank(H_M))
    A = default_rng().choice(H, 2*n, replace = False).view(GF)
    sol = solvesystem(A, GF.Ones((2*n, 1)), all_sol=True)




# %%
if __name__ == "__main__":
    '''
    Test random matrix
    '''
    n = 100
    seed = default_rng()
    s = GF.Random(n, seed=seed)
    g = 1
    seed = default_rng(seed)
    H_M = random_main_part(n, g, s, seed = seed)
    seed = default_rng(seed)
    H = add_redundancy(H_M, s, 50*len(H_M), seed=seed)
    # print(H[5])
    KMA = KMAttack(H)
    while KMA.check_d(s) == False:
        KMA.regenerate_d()
    print(KMA.check_d(s))
    print("Rank of the Gram matrix: ", KMA.get_Gs_rank(s))
    M = KMA.get_M(1000)
    S = solvesystem(M)
    print("Dimension of solution space: ", len(S))
    if len(S) < 12:
        S = solvesystem(M, all_sol=True)
    for s0 in S[:20]:
        _rank = KMA.get_Gs_rank(s0)
        if _rank == KMA.get_Gs_rank(s):
            print(np.all(s0 == s))
        print(_rank)

# %%
if __name__ == "__main__":
    '''Debug satisfying weight constraint'''
    n = 5
    seed = 0
    s = GF.Random(n, seed = default_rng(seed))
    tab = random_tableau(n, 2, s, seed = default_rng(seed))
    G = random_gram(n, 2, s, seed = default_rng(seed))
    print(tab)

    sc = Factorization(tab, s)
    print(sc.get_weight())

    E_init = sc.init_factor()
    # print(E_init.T @ E_init)
    # print(E_init)
    E = sc.satisfy_weight_constraint(E_init)
    # print(E)
    # print(E.T @ E)
    H = sc.injecting_ones(E, s = s)
    # print(H)
    H = sc.satisfy_weight_constraint(H)
    print(H.T @ H)

    # H = sc.satisfy_weight_constraint(H)

# %%
if __name__ == "__main__":
    P = np.loadtxt("./examples/challenge.xprog", dtype = int).view(GF)
    # print(GF(P))
    KMA = KMAttack(P)
    # KMA.regenerate_d()
    M = KMA.get_M(1000)
    S = solvesystem(M, all_sol=True)
    for s in S:
        _rank = KMA.get_Gs_rank(s)
        if _rank == 1:
            print("Secret found!")
            break
    H_M = GF([p for p in P if np.dot(p, s) == 1])

    H = add_redundancy(H_M, s, np.ceil(np.log(H_M.shape[1])), seed=None)
    print(len(H))

    KMA = KMAttack(H)
    while KMA.check_d(s) == False:
        KMA.regenerate_d()
    print("Correct d: ", KMA.check_d(s))
    print("Rank of the Gram matrix: ", KMA.get_Gs_rank(s))
    M = KMA.get_M(1000)
    S = solvesystem(M)
    print("Dimension of solution space: ", len(S))
    if len(S) < 10:
        S = solvesystem(M, all_sol=True)
    for s0 in S[:20]:
        _rank = KMA.get_Gs_rank(s0)
        if _rank == KMA.get_Gs_rank(s):
            print(np.all(s0 == s))
        print(_rank)



############## test redundancy ###################

# %%
def stabilizer_row_redundancy():
    '''
    Test adding row redundancy in stabilizer construction 
    '''
    # tab = np.loadtxt("./examples/4qubit_rank_1.tab", dtype = int)
    n = 70
    seed = 1
    s = GF.Random(n, seed=seed)
    g = 1
    H_M = random_main_part(n, g, s, seed = seed)
    G = random_gram(n, g, s, seed = seed)
    tab = random_tableau(n, g, s, seed = seed)
    # print(tab)
    # G = GF.Ones((n, n))
    # tab = np.hstack((G, GF.Identity(n), GF.Ones((n, 1))))
    
    sc = Factorization(tab.copy(), s)
    H_M = sc.final_factor(obf_rows=5)
    assert np.all(H_M.T @ H_M == G)
    print(rank(H_M))
    assert np.all(H_M @ s.reshape(-1, 1) == 1) # check codeword
    print(f"# rows in H_M: {len(H_M)}")
    H = add_row_redundancy(H_M, s, 0)
    print("# rows of H: ", len(H))

    attack(H, s)

    # try another attack
    # print("Rank of H_M:", rank(H_M))
    # A = default_rng().choice(H, 2*n, replace = False).view(GF)
    # sol = solvesystem(A, GF.Ones((2*n, 1)), all_sol=True)



def stabilizer_col_redundancy():
    '''
    Test adding row redundancy in stabilizer construction 
    '''
    # tab = np.loadtxt("./examples/4qubit_rank_1.tab", dtype = int)
    n = 40
    seed = 1
    s = GF.Random(n, seed=seed)
    g = 1
    H_M = random_main_part(n, g, s, seed = seed)
    G = random_gram(n, g, s, seed = seed)
    tab = random_tableau(n, g, s, seed = seed)
    # print(tab)
    # G = GF.Ones((n, n))
    # tab = np.hstack((G, GF.Identity(n), GF.Ones((n, 1))))
    
    sc = Factorization(tab.copy(), s) # initialize the construction
    H_M = sc.final_factor(obf_rows=5)
    assert np.all(H_M.T @ H_M == G)
    print("rank of H_M:", rank(H_M))
    assert np.all(H_M @ s.reshape(-1, 1) == 1) # check codeword
    H_M, s = add_col_redundancy(H_M, s, H_M.shape[0] - H_M.shape[1])
    print(f"shape H_M: {H_M.shape}")
    H = add_row_redundancy(H_M, s, 1.5*(H_M.shape[1] - rank(H_M)))
    print("shape of H: ", H.shape)

    attack(H, s)


# QRC_column_redundancy()
# stabilizer_row_redundancy()
stabilizer_col_redundancy()

# %%
n = 50
seed = 3
s = GF.Random(n, seed=seed)
g = 1
H_M = random_main_part(n, g, s, seed = seed)
G = random_gram(n, g, s, seed = seed)
tab = random_tableau(n, g, s, seed = seed)
# print(tab)
# G = GF.Ones((n, n))
# tab = np.hstack((G, GF.Identity(n), GF.Ones((n, 1))))

sc = Factorization(tab.copy(), s)
H_M = sc.final_factor(obf_rows=5)
assert np.all(H_M.T @ H_M == G)
print(rank(H_M))

d = GF.Random((n, 1))
e = GF.Random((n, 1))
print(np.mean((H_M @ d).view(np.ndarray) + (H_M @ e).view(np.ndarray) == 2))