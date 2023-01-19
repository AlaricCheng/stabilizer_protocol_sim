# %%
from typing import Union, Optional, List
import numpy as np
from numpy.random import default_rng
import galois
import os, sys
import re

GF = galois.GF(2)

# %%
bias = lambda g: (1+2**(-g/2))/2

def load_data(fname):
    H = np.loadtxt(fname, dtype=int).view(GF)
    with open(fname, "r") as f:
        for line in f:
            if line[0] == "#" and "s" in line:
                s = GF([int(c) for c in re.findall(r"\d+", line)]) 
                return H, s
    return H

class HiddenPrints:
    '''
    suppress the printing in function calls
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def wrap_seed(seed: Optional[Union[int, np.random.Generator]]):
    '''
    Convert seed into a np.random.Generator
    '''
    if seed is None or type(seed) == int:
        rng = default_rng(seed)
    elif type(seed) == np.random.Generator:
        rng = seed
    else:
        raise ValueError("Seed must be an integer, a numpy.random.Generator or None.")
    return rng

def int2bin(n, length):
    '''
    Convert the integer `n` into binary representation of length `length`
    '''
    bin_list = list(bin(n)[2:].zfill(length))
    return GF(bin_list)

def solvesystem(
    A: Union['np.ndarray', 'galois.FieldArray'], 
    b: Union['np.ndarray', 'galois.FieldArray', None] = None, 
    all_sol: bool = False
):
    '''
    Solve A x = b over GF(2)
    '''
    assert len(A) != 0, "Empty matrix!"
    A = A.view(GF)
    n = A.shape[1] # n_cols
    null = A.null_space() # basis matrix of null space
    if all_sol == True:
        complete_set = [int2bin(i, len(null)) @ null for i in range(1, 2**(len(null)))] # # linear combination of all vectors in null space

    if (b is None) or np.all(b == 0): # homogeneous equations
        assert np.all(A @ null.transpose() == 0)
        if all_sol == True:
            return GF(complete_set)
        return null
    else: # b != 0
        assert A.shape[0] == b.shape[0], "Inconsistent shapes"
        Ab = np.hstack((A, b.reshape(-1, 1)))
        Ab_reduced = Ab.row_reduce() # Gaussian elimination
        A_rd = Ab_reduced[:, :n]
        b_rd = Ab_reduced[:, n]
        free_var = np.all(A_rd == 0, axis=1).nonzero()[0] # indices for free variables
        if np.all(b_rd[free_var] == 0): 
            s_sol = specific_sol(A_rd, b_rd)
            if all_sol == True:
                if len(null) == 0: # case: A is full column rank
                    return s_sol
                return np.vstack((s_sol, GF(complete_set) + s_sol))
            return s_sol, null
        else: # case: no solution
            return [] 

def specific_sol(A, b):
    '''
    [A, b] should be in the row reduced form
    '''
    m, n = A.shape
    s_sol = GF.Zeros(n)
    for idx in range(m):
        A_i = A[-1+idx]
        b_i = b[-1+idx]
        nonzero_idx = A_i.nonzero()[0]
        if b_i == 0:
            s_sol[nonzero_idx] = 0
        elif b_i == 1 and len(nonzero_idx) == 1:
            s_sol[nonzero_idx] = 1
        elif b_i == 1 and len(nonzero_idx) > 1:
            s_sol[nonzero_idx[0]] = 1
    return s_sol

def rank(A):
    '''
    Return the rank of a matrix A.
    '''
    return np.sum(~np.all(A.row_reduce() == 0, axis=1))

def remove_all_zero_rows(A: galois.GF(2)):
    '''
    remove the all-zero rows from A
    '''
    bool_idx = np.any(A != 0, axis = 1)
    return A[bool_idx]

def KF_partition(A: galois.GF(2)):
    '''
    Given a matrix A, reorder it into (K \\ F) s.t. K.1 = 0
    '''
    A = remove_all_zero_rows(A)
    ind_row = A[0:1]
    for row in A[1:]:
        tmp = np.append(ind_row, row.reshape(1, -1), axis = 0)
        if len(tmp.row_space()) == len(ind_row):
            x = solvesystem(ind_row.T, row)[0]
            y = GF.Zeros(len(A) - len(x))
            y[0] = 1
            x = np.append(x, y)
            break
        ind_row = tmp
    K = A[x == 1]
    F = A[x == 0]

    return K, F

def lempel_sequence(E: galois.GF(2), n_rows = None) -> List[galois.GF(2)]:
    '''
    Find the Lempel sequence. If n_rows is not None, return the factor whose number of rows is closest to n_rows.
    '''
    n = E.shape[1]
    diff_0 = E.shape[0]
    seq = []
    for _ in range(len(E)):
        E = remove_all_zero_rows(E)
        if n_rows is not None:
            if E.shape[0] == n_rows or abs(E.shape[0] - n_rows) >= diff_0:
                return E
            diff_0 = abs(n_rows - E.shape[0]) # update the difference
        seq.append(E)
        if len(E.row_space()) == len(E): # if all rows in E are independent
            return seq
        elif len(E) == 3:
            K, F = KF_partition(E)
            if len(F) == 1:
                E = F
                continue
            elif len(F) == 0:
                return seq
        
        K, F = KF_partition(E)
        if len(K) == 2:
            E = F
            continue
        elif len(K) % 2 == 0:
            Z = K
        else:
            Z = np.vstack((K, GF.Zeros((1, n))))
        if len(F) == 0:
            F = GF.Zeros((1, n))
        x = (F[0] + Z[0]).reshape(1, -1)
        Z_tilde = Z + GF.Ones((len(Z), 1)) @ x
        E = np.vstack((Z_tilde[1:], F[1:]))
        


