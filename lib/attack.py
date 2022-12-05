# %%
from typing import Optional, Union
import numpy as np
from numpy.random import default_rng
import galois
from lib.utils import solvesystem, rank, wrap_seed, int2bin

GF = galois.GF(2)

class LinearityAttack:
    '''
    The inner-product attack algorithm and its analysis.
    '''
    def __init__(
        self, 
        P: 'galois.GF(2)', 
        random_state: Optional[Union[int, np.random.Generator]] = None
    ):
        self.P = P
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
        idx = self.P @ d.reshape(-1, 1)
        P_s = self.P[idx.flatten().view(np.ndarray).astype(bool)]
        return P_s

    def check_d(self, s):
        '''
        Check whether d is in the null space of G_s. 
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
            P_de = P_d[idx.flatten().view(np.ndarray).astype(bool)]
            if len(P_de) == 0:
                continue
            M.append(np.sum(P_de, axis = 0))
        return GF(M)

    def print_candidate_secret(self, S, threshold = 5, print_rank = False):
        '''
        Add a rank checking step.
        '''
        candidate = []
        rank = []
        for s in S:
            rank.append(self.get_Gs_rank(s))
            if self.get_Gs_rank(s) <= threshold:
                candidate.append(s)
        if print_rank == True:
            print(rank)
        return GF(candidate)


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

    def extract_secret_original(
        self, 
        l: int, 
        budget: Optional[int] = None
    ):
        '''
        The original extract secret subroutine of KM if `budget` is set to be None, which finds at most one secret
        Args:
            l (int): number of linear equations
            budget (int): the maximum number of checking secrets
        '''
        count = 0
        for _ in range(10):
            self.regenerate_d()
            M = self.get_M(l)
            if M is None:
                continue

            ker_M = solvesystem(M)
            if len(ker_M) == 0:
                continue
            for i in range(2**(len(ker_M))):
                y = int2bin(i, len(ker_M))
                s = y.reshape(1, -1) @ ker_M
                if self.check_weight(s):
                    return s
                count += 1
                if budget is not None and count >= budget:
                    return s
        return s

    def classical_sampling_original(
        self, 
        n_samples: int, 
        budget: Optional[int] = None,
        verbose: bool = False
    ):
        '''
        Generate the samples to pass the verifier's test
        '''
        s = self.extract_secret_original(10*self.P.shape[0], budget=budget)
        return classical_samp_same_bias(s, 0.854, n_samples, verbose = verbose)

    def extract_secret_enhanced(
        self,
        l: int,
        budget: Optional[int] = 2**15
    ):
        '''
        The enhanced extract secret subroutine for QRC-based construction, which finds a candidate set of secrets
        Args:
            l (int): number of linear equations
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
            for i in range(2**(len(ker_M))):
                y = int2bin(i, len(ker_M))
                s = y.reshape(1, -1) @ ker_M
                if self.check_weight(s):
                    S.append(s)
                count += 1
                if count >= budget and len(S) == 0:
                    return s
        return np.unique(np.vstack(S), axis = 0)

    def classical_sampling_enhanced(
        self,
        n_samples: int, 
        budget: Optional[int] = 2**15,
        verbose: bool = False
    ):
        '''
        Generate the samples to pass the verifier's test
        '''
        S = self.extract_secret_enhanced(10*self.P.shape[0], budget=budget)
        return classical_samp_same_bias(S, 0.854, n_samples, verbose = verbose)

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
    n_samples: int
):
    '''
    Classical sampling for the case where the candidates all have the same associated bias. `beta` is in the ascending order.
    '''
    S = GF(S)
    beta = np.array(beta)
    if len(S.shape) == 1:
        raise ValueError("There is only one candidate; use `classical_samp_same_bias` instead")
    beta_unique = np.unique(beta)
    prob = []
    for b1, b2 in zip(beta_unique, beta_unique[1:]):
        prob.append(b2 - b1)
    prob = [beta_unique[0]] + prob + [1 - beta_unique[-1]] # (beta_1, beta_2 - beta_1, ..., beta_t - beta_(t-1), 1 - beta_t)

    # construct the b's
    y_set = [GF.Zeros(S.shape[1])]
    for ele in beta_unique:
        b = (beta <= ele).astype(int).view(GF)
        spe_sol, _ = solvesystem(S, b)
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




