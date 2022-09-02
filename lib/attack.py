# %%
import numpy as np
from numpy.random import default_rng
import galois

GF = galois.GF(2)


class IPAttack:
    '''
    The inner-product attack algorithm and its analysis.
    '''
    def __init__(self, P):
        self.P = P
        self.n_col = P.shape[1]
        self.rng = default_rng()
        self.d = GF.Random(self.n_col, seed = self.rng)

    def set_rng(self, rng):
        self.rng = rng

    def get_d(self):
        return self.d

    def regenerate_d(self, seed = None):
        self.d = GF.Random(self.n_col, seed = seed)

    def set_d(self, d):
        self.d = GF(d)

    def get_P_s(self, s):
        '''
        Get P_s
        '''
        s = GF(s)
        idx = self.P @ s.reshape(-1, 1)
        P_s = self.P[idx.flatten().view(np.ndarray).astype(bool)]
        return P_s

    def check_d(self, s):
        '''
        Check whether d is in the null space of G_s
        '''
        P_s = self.get_P_s(s)
        G_s = P_s.T @ P_s
        return np.all(G_s @ self.d == 0)

    def get_Gs_rank(self, s):
        '''
        Get the rank of G_s
        '''
        P_s = self.get_P_s(s)
        G_s = P_s.T @ P_s
        nullity = len(G_s.null_space())
        return self.n_col - nullity

    def get_M(self, num_e):
        '''
        Generate the linear system M
        '''
        idx = self.P @ self.d.reshape(-1, 1)
        P_d = self.P[idx.flatten().view(np.ndarray).astype(bool)]
        # P_d = GF([p for p in self.P if np.dot(p, self.d) == 1])
        if len(P_d) == 0:
            return []
        M = []
        for _ in range(num_e):
            e = self.rng.choice(2, size = self.n_col).view(GF)
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


class KMAttack(IPAttack):
    '''
    Kahanamoku-Meyer's attack algorithm
    '''
    def check_weight(self, s):
        '''Check whether Hamming weight of each columns is 0 or 3'''
        P_s = self.get_P_s(s)
        max_iter = 40 # max number of codewords to be checked
        for _ in range(max_iter):
            x = self.rng.choice(2, size = self.n_col).view(GF)
            c = P_s @ x.reshape(-1, 1) # a random codeword
            weight = np.sum(c.view(np.ndarray)) % 4
            if weight != 0 and weight != 3:
                return False
        return True

    def print_candidate_secret(self, S, print_rank = False):
        '''Print all candidate secrets that satisfy the weight constraint'''
        candidate = []
        rank = []
        for s in S:
            rank.append(self.get_Gs_rank(s))
            if self.check_weight(s):
                candidate.append(s)
        if print_rank == True:
            print(rank)
        return GF(candidate)
