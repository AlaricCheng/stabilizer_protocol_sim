# %%
from typing import Union
import numpy as np
from numpy.random import default_rng
import galois
import os, sys

GF = galois.GF(2)

# %%
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

def wrap_seed(seed):
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

