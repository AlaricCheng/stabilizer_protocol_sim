# %%
from typing import Optional, Union
import numpy as np
from numpy.random import default_rng
import galois
from lib.utils import solvesystem, rank, wrap_seed, int2bin, get_H_s, iter_column_space, random_codeword, hamming_weight, get_R_s, get_D_space, check_D_doubly_even
from lib.hypothesis import bias

GF = galois.GF(2)


def property_check(H, s_i, rank_thres = 5):
    """
    check whether rank(H_{s_i}^T H_{s_i}) <= rank_thres, and whether D_{s_i} is doubly even
    """
    H_si = get_H_s(H, s_i)
    g = rank(H_si.T @ H_si)

    if g <= rank_thres:
        D = get_D_space(H_si)
        if D is None:
            return True
        return check_D_doubly_even(D)
    
    return False


def qrc_check(H, s_i):
    '''
    Check whether the code C_{s_i} is QRC by 
    checking whether Hamming weight of each codeword is 0 or 3
    '''
    H_si = get_H_s(H, s_i)
    max_iter = 40 # max number of codewords to be checked
    for _ in range(max_iter):
        c = random_codeword(H_si)
        weight = hamming_weight(c) % 4
        if weight != 0 and weight != 3:
            return False
    return True


def extract_one_secret(
        H: "galois.FieldArray", 
        g_thres = 5, 
        max_iter: int = 2**15, 
        check = "rank",
        seed = None
    ):
    rng = wrap_seed(seed)
    count = 0

    while count < max_iter:
        d = GF.Random(H.shape[1], seed = rng)
        H_d = get_H_s(H, d)
        G_d = H_d.T @ H_d
        ker_Gd = G_d.null_space()
        print("Dimension of ker(G_d): ", len(ker_Gd))
        ker_Gd_space = iter_column_space(ker_Gd.T)

        for s_i in ker_Gd_space:
            count += 1
            check_res = (check == "rank" and property_check(H, s_i, g_thres)) or (check == "qrc" and qrc_check(H, s_i))
            if check_res:
                return s_i, count 
            if count >= max_iter:
                break

    return None, max_iter


def naive_sampling(H, s, n_samples):
    """
    Output x s.t. x.s = 0 with probability beta_s
    and x.s = 1 w.p. 1 - beta_s
    """
    beta_s = bias(H, s)
    s = s.reshape(1, -1)
    for _ in range(10):
        x = GF.Random((1, H.shape[1]))
        if np.inner(x, s)[0, 0] == 1:
            break
    
    s_kernel = s.null_space()
    X = []
    for _ in range(n_samples):
        coin = default_rng().choice(2, p = (beta_s, 1-beta_s))
        x_0 = random_codeword(s_kernel.T).T
        if coin == 0:
            X.append(x_0)
        else:
            X.append(x_0 + x)
    
    return np.vstack(X)


def sampling_with_H(H, s, n_samples):
    """
    Output x from row space of R_s with probability beta_s
    and from row space of H_s and satisfying x.s = 1 w.p. 1 - beta_s
    """
    H_s = get_H_s(H, s)
    R_s = get_R_s(H, s)
    beta_s = bias(H, s)
    s = s.reshape(1, -1)
    
    X = []
    for _ in range(n_samples):
        coin = default_rng().choice(2, p = (beta_s, 1-beta_s))
        if coin == 0:
            x_0 = random_codeword(R_s.row_space().T).T
            X.append(x_0)
        else:
            for _ in range(10):
                x_1 = random_codeword(H_s.row_space().T).T
                if np.inner(x_1, s)[0, 0] == 1:
                    X.append(x_1)
                    break
    
    return np.vstack(X)


def classical_samp_same_bias(
    S: galois.GF(2), 
    beta: float, 
    n_samples: int,
    verbose: bool = False
):
    '''
    Classical sampling for the case where the candidates all have the same associated bias.
    '''
    S = GF(S)
    if len(S.shape) == 1:
        S = S.reshape(1, -1)
    if verbose:
        print("candidate secret:\n", S)
    spe_sol, basis = solvesystem(S, GF.Ones((len(S), 1)))
    spe_sol = spe_sol.reshape(1, -1)
    X = []
    while len(X) < n_samples:
        coeff = GF.Random((1, len(basis)))
        coin = default_rng().choice(2, p = (beta, 1-beta))
        x = coeff @ basis
        if coin == 0:
            X.append(x)
        else:
            X.append(x + spe_sol)
    
    return np.vstack(X)


def classical_samp_diff_bias(
    S: galois.GF(2), 
    beta: list, 
    n_samples: int,
    verbose: bool = False
):
    '''
    Classical sampling for the case where the candidates all have the same associated bias. `beta` is in the ascending order.
    '''
    S = GF(S)
    beta = np.array(beta)
    if len(S.shape) == 1:
        return classical_samp_same_bias(S, beta[0], n_samples, verbose = verbose)
    if verbose:
        print("candidate secret:\n", S)
    beta_unique = np.unique(beta)
    prob = []
    for b1, b2 in zip(beta_unique, beta_unique[1:]):
        prob.append(b2 - b1)
    prob = [beta_unique[0]] + prob + [1 - beta_unique[-1]] # (beta_1, beta_2 - beta_1, ..., beta_t - beta_(t-1), 1 - beta_t)

    # construct the b's
    y_set = [GF.Zeros(S.shape[1])]
    for ele in beta_unique:
        b = (beta <= ele).astype(int).view(GF)
        try:
            spe_sol, _ = solvesystem(S, b)
        except ValueError: # no solution
            return classical_samp_same_bias(S[-1], beta[-1], n_samples, verbose = verbose)
        y_set.append(spe_sol)
    y_set = GF(y_set)
    basis = solvesystem(S, GF.Zeros((len(S), 1)))
    # classical sampling
    X = []
    while len(X) < n_samples:
        coeff = GF.Random((1, len(basis)))
        x = coeff @ basis
        i = default_rng().choice(len(prob), p = prob)
        X.append((x + y_set)[i])
        
    return np.vstack(X)




