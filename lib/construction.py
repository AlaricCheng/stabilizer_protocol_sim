from typing import Union
import numpy as np
from numpy.random import default_rng
import galois
from .utils import solvesystem, wrap_seed, rank
import itertools

GF = galois.GF(2)

__all__ = [
    "random_main_part", "random_gram", "random_tableau",
    "add_row_redundancy", "add_col_redundancy", 
    "Factorization", "QRCConstruction", "generate_QRC_instance"
]

def random_main_part(n, g, s, seed = None):
    '''
    Randomly sample g rows that have inner products one with the secret s. 
    Args:
        n (int): number of qubits
        g (int): number of rows
        s (galois.FieldArray): secret vector
        seed (int | np.random.Generator): seed for random sampling
    Return:
        A random main part with g rows.
    '''
    assert len(s) == n, "Inconsistent shapes"
    H = []
    rng = wrap_seed(seed)
    while len(H) < g:
        # seed = int(rng.random() * 10**5)
        row = GF.Random(n, seed = rng)
        if np.dot(row, s) == 1:
            H.append(row)
    H = GF(H)
    return H

def random_gram(n, g, s, seed = None):
    '''
    Generate a random Gram matrix with rank <= g. It is generated by first sampling g rows that are not orthogonal to s, to form H. Then, G = H^T \cdot H
    Args:
        n (int): number of qubits
        g (int): number of rows
        s (galois.FieldArray): secret vector
        seed (int | np.random.Generator): seed for random sampling
    Return:
        A random Gram matrix generated by G = H^T \cdot H
    '''
    rng = wrap_seed(seed)
    H = random_main_part(n, g, s, seed = rng)
    G = H.T @ H
    return G

def random_tableau(n, g, s, seed = None):
    '''
    Generate a random stabilizer tableau whose X-part is the Gram matrix, and Z-part is identity. The Gram matrix has rank <= g. 
    Args:
        n (int): number of qubits
        g (int): number of rows
        s (galois.FieldArray): secret vector
        seed (int | np.random.Generator): seed for random sampling
    Return:
        A random stabilizer tableau. 
    '''
    rng = wrap_seed(seed)
    G = random_gram(n, g, s, seed = rng)
    x = GF.Random((n, 1), seed = rng)
    r = G @ x # to ensure that the overlap is not zero.
    stab_tab = np.hstack((G, GF.Identity(n), r))
    return stab_tab

def add_row_redundancy(H, s, size, seed = None):
    '''
    Given the main part and secret, append redundant rows (number of rows given by size).
    Args:
        H (galois.FieldArray): binary matrix to be appended redundant rows
        s (galois.FieldArray): secret vector
        size (int): number of redundant rows
        seed (int | np.random.Generator): seed for random sampling
    '''
    if size == 0:
        return H
    H_R = []
    n = H.shape[1]
    rng = wrap_seed(seed)
    while len(H_R) < size:
        row = GF.Random(n, seed = rng)
        if np.dot(row, s) == 0 and np.any(row != 0): # exclude all-zero rows
            H_R.append(row)
    return np.append(H, H_R, axis = 0)

def add_col_redundancy(H_M, s, size, seed = None):
    '''
    Given the main part and the secret, append random codewords to the columns of H_M. 
    Args:
        H_M (galois.FieldArray): binary matrix to be appended redundant columns
        s (galois.FieldArray): secret vector
        size (int): number of redundant rows
        seed (int | np.random.Generator): seed for random sampling
    Return:
        Tuple(H_M, s)
    '''
    ext_col = []
    n = H_M.shape[1]
    rng = wrap_seed(seed)
    while len(ext_col) < size:
        x = GF.Random((n, 1), seed = rng)
        codeword = H_M @ x # random linear combination of cols in H_M
        ext_col.append(codeword)

    ext_col: galois.FieldArray = np.hstack(ext_col)
    s_prime = ext_col.null_space()
    if len(s_prime) == 0:
        s_prime = GF.Zeros(size)
    else:
        s_prime = default_rng().choice(s_prime)

    new_H_M = np.hstack((H_M, ext_col)) # append random cols to the right of H_M
    new_s = np.append(s, s_prime)
    return new_H_M, new_s


class Factorization:
    def __init__(
        self, 
        tab: 'galois.FieldArray', 
        s: Union['galois.FieldArray', None] = None
    ):
        '''
        Given a stabilizer tableau, return a factorization of the Gram matrix satisfying the weight and codeword constraints.
        '''
        self.tab = tab.copy()
        self.n = len(self.tab)
        G = tab[:, :self.n]
        assert np.all(G == G.T), "G must be symmetric"
        self.G = G 
        self.rng = default_rng()
        if s is None:
            s = self.self_consistent_eqn(one_sol=True) 
        self.s = s

    def set_rng(self, seed):
        self.rng = wrap_seed(seed)

    def get_weight(self): # TODO
        '''
        Get weight constraint from the stabilizer tableau.
        '''
        weight_dict = {"00": 0, "01": 1, "10": 2, "11": 3}
        weights = []
        r = self.tab[:, 2*self.n]
        for i in range(self.n):
            weight = f"{r[i]}" + f"{self.G[i, i]}"
            weights.append(weight_dict[weight])
        return np.array(weights)

    def forward_evolution(self, row):
        '''
        Evolve the stabilizer tableau after applying e^{i pi X_p /4}
        '''
        indices = row.nonzero()[0]
        for idx in indices: # update phase
            if self.tab[idx, idx] == 1:
                self.tab[idx, 2*self.n] += GF(1)
        for idx in itertools.product(indices, repeat = 2): # update gram matrix
            self.tab[idx] += GF(1)
        self.__init__(self.tab, self.s)

    def backward_evolution(self, row):
        '''
        Evolve the stabilizer tableau after applying e^{-i pi X_p /4}
        '''
        indices = row.nonzero()[0]
        for idx in indices: # update phase
            if self.tab[idx, idx] == 0:
                self.tab[idx, 2*self.n] += GF(1)
        for idx in itertools.product(indices, repeat = 2): # update gram matrix
            self.tab[idx] += GF(1)
        self.__init__(self.tab, self.s)

    def obfuscation(self, n_rows):
        '''
        Obfuscation of stabilizer tableau
        '''
        H_obf = random_main_part(self.n, n_rows, self.s, seed = self.rng)
        for row in H_obf:
            self.backward_evolution(row)
        return H_obf

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

    def injecting_ones(self, E):
        '''
        Injection subroutine to inject the all-one codeword.
        '''
        indicator = E @ self.s.reshape(-1, 1) # indicator to separate two parts
        F = E[indicator.nonzero()[0]] # F.s = 1
        Z = E[np.where(indicator == 0)[0]] # Z.s = 0
        if len(Z) % 2 != 0:
            zeros = GF.Zeros((1, self.n))
            Z = np.append(Z, zeros, axis = 0)
        x = GF.Zeros(self.n)
        flip_idx = self.rng.choice(self.s.nonzero()[0]) # only a special case
        x[flip_idx] = 1
        ones = GF.Ones((len(Z), 1))
        Z = Z + ones @ x.reshape(1, -1)
        return np.append(F, Z, axis = 0)

    def final_factor(self, obf_rows = None):
        '''
        Combine the subroutines to generate the final factorization.
        Args
            obf_rows: for stabilizer tableau obfuscation
        '''
        if obf_rows is None:
            obf_rows = self.n
        H_obf = self.obfuscation(obf_rows)
        E_init = self.init_factor()
        E = self.satisfy_weight_constraint(E_init)
        H = self.injecting_ones(E)
        H = self.satisfy_weight_constraint(H)
        if obf_rows == 0:
            return H
        else:
            return np.vstack((H_obf, H))


class QRCConstruction:
    def __init__(self, q):
        # a valid q can be obtained from lib.construction.q_helper
        assert (q+1)%8 == 0, "(q + 1) must divide 8"
        self.q = q # size parameter
        self.n = int((q+3)/2) # num of qubits
        self.s = np.append([1], GF.Zeros(self.n - 1)) # initial secret
        self.P_s = self.init_main() # initial main part

    def quad_res(self):
        '''
        Generate the list of quadratic residues modulo 1.
        '''
        QRs = []
        for m in range(self.q):
            QRs.append(m**2% self.q)
        QRs.pop(0)
        return list(set(QRs))

    def init_main(self):
        '''
        Generate initial main part
        '''
        P_s = GF.Zeros((self.q, self.n))
        P_s[:, 0] = 1
        QRs = self.quad_res() # the list of quadratic residues
        for col in range(self.n-1):
            for qr in QRs:
                P_s[(qr - 1 + col)%self.q, col+1] = 1
        return P_s

    def ColAdd(self, i, j):
        '''
        Add the j-th column of P_s to the i-th column, and add the i-th element of s to the j-th element.
        '''
        s_i = self.s[i]
        s_j = self.s[j]
        s_j = s_i + s_j
        self.s[j] = s_j

        P_i = self.P_s[:, i]
        P_j = self.P_s[:, j]
        P_i = P_i + P_j
        self.P_s[:, i] = P_i

    def obfuscation(self, times, seed = None):
        '''
        Do column operations on P_s and s.
        '''
        rng = wrap_seed(seed)
        for _ in range(times):
            i, j = rng.choice(self.n, size = 2, replace = False)
            self.ColAdd(i, j)

def is_prime(n):
    if n % 2 == 0 and n > 2: 
        return False
    return all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2))

def q_helper(N):
    '''
    Return the list of valid q smaller than N for QRC.
    '''
    assert type(N) == int, "N must be integer"
    a = np.arange(7, N, 8)
    foo = np.vectorize(is_prime)
    pbools = foo(a)
    q_list = np.extract(pbools, a)
    return q_list

    # [7, 23, 31, 47, 71, 79, 103, 127, 151, 167, 191, 199, 223,  239, 263, 271, 311, 359, 367, 383, 431, 439, 463, 479, 487, 503,  599, 607, 631, 647, 719, 727, 743, 751, 823, 839, 863, 887, 911,  919, 967, 983, 991]


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
    QRC = QRCConstruction(q) # initialization
    QRC.obfuscation(1000)
    H_M = QRC.P_s
    s = QRC.s
    H_M, s = add_col_redundancy(H_M, s, rd_col)
    H = add_row_redundancy(H_M, s, rd_row)
    print("rank of H_M:", rank(H_M), "\trank of H:", rank(H), "\tshape of H:", H.shape)

    return H, s