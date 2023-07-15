# %%
from typing import Union, Optional
import numpy as np
from numpy.random import default_rng
import galois
import os, sys
import re

GF = galois.GF(2)

bias = lambda g: (1+2**(-g/2))/2

def load_data(fname):
    H = np.loadtxt(fname, dtype=int).view(GF)
    with open(fname, "r") as f:
        for line in f:
            if line[0] == "#" and "s" in line:
                s = GF([int(c) for c in re.findall(r"\d+", line)]) 
                return H, s
    return H

def hamming_weight(x: "galois.FieldArray"):
    """
    Hamming weight of x
    """
    return np.sum(x.view(np.ndarray))
    
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
    null = A.null_space() # basis matrix of null space of rows of A
    if all_sol == True:
        complete_set = [int2bin(i, len(null)) @ null for i in range(1, 2**(len(null)))] # # linear combination of all vectors in null space

    if (b is None) or np.all(b == 0): # homogeneous equations
        assert np.all(A @ null.transpose() == 0)
        if all_sol == True:
            return GF(complete_set)
        return null
    else: # b != 0
        assert A.shape[0] == len(b), f"Inconsistent shapes, {A.shape[0]} and {len(b)}"
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

def check_element(C, x):
    """
    Check whether x is in the column space of C
    """
    if solvesystem(C, x) == []:
        return False
    else:
        return True
    
def fix_basis(A: 'galois.GF(2)', basis: 'galois.GF(2)'):
    """
    Change the first k columns of basis to be the columns of A.
    """
    for b in basis.T:
        if not check_element(A, b):
            A = np.concatenate((A, b.reshape(-1, 1)), axis = 1)
    
    return A

def sample_column_space(basis: 'galois.GF(2)', seed = None):
    """
    Given a basis in the form of a matrix, sample a random vector from its column space.
    """
    n = basis.shape[1]
    for _ in range(20):
        x = GF.Random((n, 1), seed = seed)
        if hamming_weight(x) != 0:

            return basis @ x

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


def rand_inv_mat(n, seed = None):
    '''
    Generate a random invertible matrix of size n x n
    '''
    rng = wrap_seed(seed)
    A = GF.Random((n, n), seed = rng)
    while rank(A) != n:
        A = GF.Random((n, n), seed = rng)

    return A


def get_H_s(H: "galois.FieldArray", s: 'galois.FieldArray'):
        '''
        Get H_s by deleting rows that are orthogonal to s
        '''
        idx = H @ s.reshape(-1, 1)
        H_s = H[(idx == 1).flatten()]

        return H_s


def get_R_s(H: "galois.FieldArray", s: 'galois.FieldArray'):
        '''
        Get R_s by deleting rows that are not orthogonal to s
        '''
        idx = H @ s.reshape(-1, 1)
        R_s = H[(idx == 0).flatten()]

        return R_s