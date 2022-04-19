# %%
import numpy as np
from numpy.random import default_rng
import galois
import stim
from .utils import solvesystem

GF = galois.GF(2)

# %%
def random_gram(n, g, s):
    '''
    Generate a random Gram matrix with rank <= g. It is generated by first sampling g rows that are not orthogonal to s, to form H. Then, G = H^T.H
    '''
    assert len(s) == n, "Inconsistent shapes"
    H = []
    while len(H) < g:
        row = GF.Random(n)
        if np.dot(row, s) == 1:
            H.append(row)
    H = GF(H)
    G = H.T @ H
    return G


def random_tableau(n, g, s):
    '''
    Generate a random stabilizer tableau whose X-part is the Gram matrix, and Z-part is identity. The Gram matrix has rank <= g. 
    '''
    G = random_gram(n, g, s)
    x = GF.Random((n, 1))
    r = G @ x
    stab_tab = np.hstack((G, GF.Identity(n), r))
    return stab_tab


def add_redundancy(H, s, size):
    '''
    Given the main part and secret, append redundant rows (number of rows given by size).
    '''
    H_R = []
    n = H.shape[1]
    while len(H_R) < size:
        row = GF.Random(n)
        if np.dot(row, s) == 0 and np.any(s != 0):
            H_R.append(row)
    return np.append(H, H_R, axis = 0)



# %%
class Factorization:
    def __init__(self, tab, s = None):
        '''
        Given a stabilizer tableau, return a factorization of the Gram matrix satisfying the weight and codeword constraints.
        '''
        self.tab = tab #.astype(int).view(GF)
        self.s = s
        self.n = len(self.tab)
        G = tab[:, 0:self.n]
        assert np.all(G == G.T), "G must be symmetric"
        self.G = G #.astype(int).view(GF)
        self.rng = default_rng()

    def get_weight(self):
        '''
        Get weight constraint from the stabilizer tableau.
        '''
        weight_dict = {"00": 0, "01": 1, "10": 2, "11": 3}
        weights = []
        r = self.tab[:, 2*self.n]
        for i in range(self.n):
            weight = f"{self.G[i, i]}" + f"{r[i]}"
            weights.append(weight_dict[weight])
        return np.array(weights)
        
    def init_factor(self):
        '''
        Construct the initial factorization, based on [Lempel 75]. 
        '''
        row_sum = np.sum(self.G, axis = 1)
        N1 = np.nonzero(row_sum)[0] # idx for odd-parity rows

        E = GF.Zeros((len(N1), self.n))
        for i in range(len(N1)):
            E[i, N1[i]] = 1 # part corresponding to N1
        
        for i in range(self.n): # part corresponding to N2
            for j in range(i+1, self.n):
                if self.G[i, j] == 1:
                    tmp = GF.Zeros((1, self.n))
                    tmp[0, [i, j]] = 1, 1
                    E = np.append(E, tmp, axis = 0)
        return E

    def satisfy_weight_constraint(self, E):
        '''
        Make the factor satisfy the weight constraint.
        '''
        weights = np.sum(E.view(np.ndarray), axis = 0) % 4 # weights of columns of E mod 4
        weight_diff = (self.get_weight() - weights) % 4
        for idx, w in enumerate(weight_diff):
            if w == 0:
                continue
            rows = GF.Zeros((w, self.n))
            rows[:, idx] = GF.Ones(w)
            E = np.append(E, rows, axis = 0)
        return E
    
    def self_consistent_eqn(self, one_sol = False):
        '''
        Generte solutions of self-consistent equation G.s = (|c_1|, ..., |c_n|)^T. 
        If one_sol = True, output a random vector from the solution space
        '''
        w = GF([self.G[i, i] for i in range(self.n)])
        candidate = solvesystem(self.G, w, all_sol=True) # solutions of G.s = w
        if one_sol == True:
            idx = self.rng.choice(len(candidate))
            return candidate[idx]
        else:
            return candidate

    def injecting_ones(self, E, s = None):
        '''
        Injection subroutine to inject the all-one codeword.
        '''
        if s is None:
            s = self.self_consistent_eqn(one_sol=True) 
        indicator = E @ s.reshape(-1, 1) # indicator to separate two parts
        F = E[indicator.nonzero()[0]] # F.s = 1
        Z = E[np.where(indicator == 0)[0]] # Z.s = 0
        if len(Z) % 2 != 0:
            zeros = GF.Zeros((1, self.n))
            Z = np.append(Z, zeros, axis = 0)
        x = GF.Zeros(self.n)
        flip_idx = self.rng.choice(s.nonzero()[0]) # only a special case
        x[flip_idx] = 1
        ones = GF.Ones((len(Z), 1))
        Z = Z + ones @ x.reshape(1, -1)
        return np.append(F, Z, axis = 0)


    def final_factor(self):
        '''
        Combine the subroutines to generate the final factorization.
        '''
        E_init = self.init_factor()
        E = self.satisfy_weight_constraint(E_init)
        H = self.injecting_ones(E, self.s)
        H = self.satisfy_weight_constraint(H)
        return H




