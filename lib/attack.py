# %%
from typing import Optional, Union
import numpy as np
from numpy.random import default_rng
import galois
from lib.utils import solvesystem, rank, wrap_seed, int2bin, bias, get_H_s, iter_column_space, random_codeword, hamming_weight, get_R_s, get_D_space

GF = galois.GF(2)


def check_D_doubly_even(D):
    """
    check whether D spans a doubly-even code
    """
    for c in D.T:
        if hamming_weight(c) % 4 != 0:
            return False
    return True


def property_check(H, s_i, rank_thres = 5):
    """
    check whether rank(H_{s_i}^T H_{s_i}) <= rank_thres
    """
    H_si = get_H_s(H, s_i)
    g = rank(H_si.T @ H_si)

    if g <= rank_thres:
        if rank(H_si) == g: # no D space
            return True
        D = get_D_space(H_si, g)
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
    H_s = get_H_s(H, s)
    beta_s = bias(rank(H_s.T @ H_s))
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
    
    return np.concatenate(X, axis = 0)


def sampling_with_H(H, s, n_samples):
    """
    Output x from row space of R_s with probability beta_s
    and from row space of H_s and satisfying x.s = 1 w.p. 1 - beta_s
    """
    H_s = get_H_s(H, s)
    R_s = get_R_s(H, s)
    beta_s = bias(rank(H_s.T @ H_s))
    s = s.reshape(1, -1)
    
    X = []
    for _ in range(n_samples):
        coin = default_rng().choice(2, p = (beta_s, 1-beta_s))
        x_0 = random_codeword(R_s.row_space().T).T
        if coin == 0:
            X.append(x_0)
        else:
            for _ in range(10):
                x_1 = random_codeword(H_s.row_space().T).T
                if np.inner(x_1, s)[0, 0] == 1:
                    X.append(x_1)
                    break
    
    return np.concatenate(X, axis = 0)


class LinearityAttack:
    '''
    Linearity attack for general theta = pi/8 IQP circuits.
    '''
    def __init__(
        self, 
        P: 'galois.GF(2)', 
        random_state: Optional[Union[int, np.random.Generator]] = None
    ):
        self.H = P
        self.n_col = P.shape[1]
        self.rng = wrap_seed(random_state)
        self.d = GF.Random(self.n_col, seed = self.rng)

    def get_d(self):
        return self.d

    def regenerate_d(self):
        self.d = GF.Random(self.n_col, seed = self.rng)

    def set_d(self, d):
        self.d = GF(d)

    def get_P_d(self, d: 'galois.GF(2)'):
        '''
        Get P_d by deleting rows that are orthogonal to d
        '''
        d = GF(d)
        idx = self.H @ d.reshape(-1, 1)
        P_s = self.H[(idx == 1).flatten()]
        return P_s

    def check_d(self, s):
        '''
        Check whether d is in the kernel of G_s. 
        This is only for testing purpose, and not for the actual attack
        '''
        P_s = self.get_P_d(s)
        G_s = P_s.T @ P_s
        return np.all(G_s @ self.d == 0)

    def get_Gs_rank(self, s):
        '''
        Get the rank of G_s
        '''
        P_s = self.get_P_d(s)
        G_s = P_s.T @ P_s
        return rank(G_s)

    def get_M(self, size):
        '''
        Generate the linear system M, which might be empty if d is not good
        '''
        P_d = self.get_P_d(self.d)
        if len(P_d) == 0:
            return
        M = []
        while len(M) < size:
            e = GF.Random(self.n_col, seed = self.rng)
            idx = P_d @ e.reshape(-1, 1)
            P_de = P_d[(idx == 1).flatten()]
            if len(P_de) == 0:
                continue
            M.append(np.sum(P_de, axis = 0))
        return GF(M)

    def extract_secret(
        self, 
        l: int,
        g_thres: int = 5,
        budget: Optional[int] = 2**15,
        verbose: bool = False
    ):
        '''
        The extract secret subroutine of the Linearity attack.
        Args:
            l (int): number of linear equations
            g_thres (int): the threshold of the rank of G_s
            budget (int): the maximum number of checking secrets
        '''
        count = 0
        S = []
        while count < budget:
            self.regenerate_d()
            M = self.get_M(l)
            if M is None:
                continue
            ker_M = solvesystem(M)
            if len(ker_M) == 0:
                continue
            if verbose:
                print("The dimension of the kernel is: ", len(ker_M))
            for i in range(1, 2**(len(ker_M))):
                y = int2bin(i, len(ker_M))
                s = y.reshape(1, -1) @ ker_M
                if self.get_Gs_rank(s) <= g_thres:
                    b = self.H @ s.reshape(-1, 1)
                    S.append(solvesystem(self.H, b, all_sol = True))
                    if verbose:
                        print(f"Found {len(S[-1])} equivalent secrets")
                count += 1
                if count >= budget:
                    break
        if len(S) == 0:
            return s, count
        S = np.unique(np.vstack(S), axis = 0).view(GF)

        return S, count

    def classical_sampling(
        self,
        n_samples: int,
        budget: Optional[int] = 2**15,
        independent_candidate: bool = True,
        g_thres: int = None,
        verbose: bool = False,
        require_count: bool = False
    ):
        '''
        Generate the samples to pass the verifier's test
        '''
        if g_thres is None:
            cor_func_list = np.array([2**(-g/2) for g in range(1, 6)])
            g_thres = np.abs(cor_func_list - 2/np.sqrt(n_samples)).argmin() + 1
        S, count = self.extract_secret(10*self.H.shape[0], g_thres, budget = budget, verbose=verbose)
        if independent_candidate:
            S = find_independent_sets(S)
        beta = []
        for s in S:
            g = self.get_Gs_rank(s)
            beta.append(bias(g))
        idx = sorted(range(len(beta)), key = lambda k: beta[k])
        S_sorted = [S[k] for k in idx]
        beta_sorted = [beta[k] for k in idx]
        
        if require_count:
            return classical_samp_diff_bias(S_sorted, beta_sorted, n_samples, verbose), count
        else:
            return classical_samp_diff_bias(S_sorted, beta_sorted, n_samples, verbose)


class QRCAttack(LinearityAttack):
    '''
    An enhanced version of Kahanamoku-Meyer's attack
    '''
    def check_weight(self, s):
        '''Check whether Hamming weight of each columns is 0 or 3'''
        if np.all(s == 0):
            return False
        P_s = self.get_P_d(s)
        max_iter = 40 # max number of codewords to be checked
        for _ in range(max_iter):
            x = GF.Random((self.n_col, 1), seed = self.rng)
            c = P_s @ x # a random codeword
            weight = np.sum(c.view(np.ndarray)) % 4
            if weight != 0 and weight != 3:
                return False
        return True

    # def extract_secret_original(
    #     self, 
    #     l: int, 
    #     budget: Optional[int] = None
    # ):
    #     '''
    #     The original extract secret subroutine of KM if `budget` is set to be None, which finds at most one secret
    #     Args:
    #         l (int): number of linear equations
    #         budget (int): the maximum number of checking secrets
    #     '''
    #     count = 0
    #     for _ in range(10):
    #         self.regenerate_d()
    #         M = self.get_M(l)
    #         if M is None:
    #             continue

    #         ker_M = solvesystem(M)
    #         if len(ker_M) == 0:
    #             continue
    #         for i in range(1, 2**(len(ker_M))):
    #             y = int2bin(i, len(ker_M))
    #             s = y.reshape(1, -1) @ ker_M
    #             if self.check_weight(s):
    #                 return s, count
    #             count += 1
    #             if budget is not None and count >= budget:
    #                 return s, count
    #     return s, count

    # def classical_sampling_original(
    #     self, 
    #     n_samples: int, 
    #     budget: Optional[int] = None,
    #     verbose: bool = False,
    #     require_count: bool = False
    # ):
    #     '''
    #     Generate the samples to pass the verifier's test
    #     '''
    #     s, count = self.extract_secret_original(10*self.H.shape[0], budget=budget)
    #     if require_count:
    #         return classical_samp_same_bias(s, 0.854, n_samples, verbose = verbose), count
    #     else:
    #         return classical_samp_same_bias(s, 0.854, n_samples, verbose = verbose)

    def extract_secret(
        self,
        l: int,
        budget: Optional[int] = 2**15,
        one_sol: bool = False,
        verbose: bool = False
    ):
        '''
        The enhanced extract secret subroutine for QRC-based construction, which finds a candidate set of secrets
        Args:
            l (int): number of linear equations
            budget (int): the maximum number of checking secrets
            one_sol (bool): whether to return only one secret
        '''
        count = 0
        S = []
        while count < budget:
            self.regenerate_d() 
            M = self.get_M(l)
            if M is None:
                continue

            ker_M = solvesystem(M)
            if len(ker_M) == 0:
                continue
            if verbose:
                print("Dimension of kernel: ", len(ker_M))
            for i in range(1, 2**(len(ker_M))):
                y = int2bin(i, len(ker_M))
                s = y.reshape(1, -1) @ ker_M
                if self.check_weight(s):
                    b = self.H @ s.reshape(-1, 1)
                    S.append(solvesystem(self.H, b, all_sol = True))
                    if one_sol:
                        return np.unique(np.vstack(S), axis = 0).view(GF), count
                count += 1
                if count >= budget:
                    break
        if len(S) == 0:
            return s, count
        return np.unique(np.vstack(S), axis = 0).view(GF), count

    def classical_sampling(
        self,
        n_samples: int, 
        budget: Optional[int] = 2**15,
        verbose: bool = False,
        require_count: bool = False,
        one_sol: bool = False
    ):
        '''
        Generate the samples to pass the verifier's test
        '''
        S, count = self.extract_secret(10*self.H.shape[0], budget=budget, one_sol = one_sol, verbose=verbose)
        S_ind = find_independent_sets(S)
        if require_count:
            return classical_samp_same_bias(S_ind, 0.854, n_samples, verbose = verbose), count
        else:
            return classical_samp_same_bias(S_ind, 0.854, n_samples, verbose = verbose)

    def classical_sampling_kernel(self, M: galois.GF(2), n_samples: int):
        '''
        Generate samples using the fact that ker(M) is a linear subspace.
        '''
        S = solvesystem(M)
        X = self.classical_sampling(S, n_samples)
        return X


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




