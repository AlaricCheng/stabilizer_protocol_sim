# %%
import numpy as np
from numpy.random import default_rng
import galois

GF = galois.GF(2)

# %%
def int2bin(n, length):
    '''Convert the integer `n` into binary representation of length `length`'''
    bin_list = list(bin(n)[2:].zfill(length))
    return GF(bin_list)


class IPAttack:
    '''
    The attack algorithm and its analysis.
    '''
    def __init__(self, P):
        self.P = P.view(GF)
        self.n_col = P.shape[1]
        self.rng = default_rng()
        self.d = self.rng.choice(2, size = self.n_col).view(GF)

    def get_d(self):
        return self.d

    def regenerate_d(self):
        self.d = self.rng.choice(2, size = self.n_col).view(GF)

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


class KMAttack(IPAttack):
    def check_weight(self, s):
        '''Check whether Hamming weight of each columns is 0 or 3'''
        P_s = self.get_P_s(s)
        max_iter = 40 
        for _ in range(max_iter):
            x = self.rng.choice(2, size = self.n_col).view(GF)
            c = P_s @ x.reshape(-1, 1)
            weight = np.sum(c.view(np.ndarray)) % 4
            if weight != 0 and weight != 3:
                return False
        return True

    def print_candidate_secret(self, S):
        '''Print all candidate secrets that satisfy the weight constraint'''
        candidate = []
        for i in range(1, 2**(len(S))):
            s = int2bin(i, len(S)) @ S
            if KMA.check_weight(s):
                candidate.append(s)
        return GF(candidate)




def solvesystem(M):
    '''
    Solve M s = 0
    '''
    assert len(M) != 0
    s = GF(M).null_space()
    assert np.all(M @ s.transpose() == 0)
    return s
    




# %%
if __name__ == "__main__":
    P = np.loadtxt("../examples/4qubit_1110.prog", dtype = int)
    # P = np.loadtxt("../examples/5qubit_QRC.prog", dtype = int)
    # P = np.loadtxt("../examples/challenge.xprog", dtype = int)
    # print(GF(P))
    KMA = KMAttack(P)
    # KMA.regenerate_d()
    M = KMA.get_M(1000)
    S = solvesystem(M)
    # for i in range(1, 2**(len(S))):
    #     s = int2bin(i, len(S)) @ S
    #     if KMA.check_weight(s):
    #         print(s)
    print(KMA.print_candidate_secret(S))
