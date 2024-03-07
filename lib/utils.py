# %%
from typing import Union, Optional
import numpy as np
from numpy.random import default_rng
import galois
import os, sys
import re
import json, uuid
import multiprocessing

GF = galois.GF(2)

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
        

def random_solution(A, b, seed = None):
    """
    Generate a random solution to A x = b
    """
    rng = wrap_seed(seed)
    try:
        s_sol, null = solvesystem(A, b)
    except ValueError:
        return None
    if len(null) == 0:
        return s_sol
    else:
        return s_sol + sample_column_space(null.T, seed = rng).flatten()
    

# def check_element(C, x):
#     """
#     Check whether x is in the column space of C
#     """
#     if solvesystem(C, x) == []:
#         return False
#     else:
#         return True

def check_element(C, x):
    """
    Check whether x is in the column space of C
    """
    tmp = np.concatenate((C, x.reshape(-1, 1)), axis = 1)
    if rank(tmp) == rank(C):
        return True
    else:
        return False

# rephrase the function `fix_basis` to speed up


def fix_basis(A: 'galois.FieldArray', basis: 'galois.FieldArray'):
    """
    Change the first k columns of basis to be the columns of A.
    """
    for b in basis.T:
        if not check_element(A, b):
            A = np.concatenate((A, b.reshape(-1, 1)), axis = 1)
    
    return A

def sample_column_space(basis: 'galois.FieldArray', seed = None, low_weight = False):
    """
    Given a basis in the form of a matrix, sample a random vector from its column space.
    """
    rng = wrap_seed(seed)
    m, n = basis.shape
    for _ in range(50):
        x = GF.Random((n, 1), seed = rng)
        if hamming_weight(x) != 0:
            c = basis @ x
            if (low_weight == True and hamming_weight(c) < 0.3 * m) or low_weight == False:
                return c


def iter_column_space(basis: "galois.FieldArray"):
    """
    Iterate over the column space of basis. Use next() to get the next vector.
    """
    n = basis.shape[1]
    for i in range(1, 2**n):
        x = int2bin(i, n).reshape(-1, 1)

        yield basis @ x


def random_codeword(basis: "galois.FieldArray", seed = None):
    """
    Return a random codeword from the column space of basis. 
    The output codeword will not be all-zero
    """
    n = basis.shape[1]
    for _ in range(20):
        x = GF.Random((n, 1), seed = seed)
        if hamming_weight(x) != 0:
            break

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
    return int(np.sum(~np.all(A.row_reduce() == 0, axis=1)))


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


def get_D_space(H: "galois.FieldArray"):
    """
    Get D = C \bigcap C^{\perp} 
    """
    G = H.T @ H
    ker = G.null_space()
    D = H @ ker.T
    return D.column_space().T


def check_D_doubly_even(D):
    """
    check whether D spans a doubly-even code
    """
    if D is not None:
        for c in D.T:
            if hamming_weight(c) % 4 != 0:
                return False
    return True

def dumpToUuid(payload):
    fileName = "log/"+str(uuid.uuid4())+".json"

    print("Dumping data to", fileName )
    with open(fileName, "w") as f:
        json.dump(payload, f) 


def estimate_distance(C: "galois.FieldArray", p = 0.5):
    """
    Estimate the minimum distance of a code generated by C
    """    
    C = C.column_space().T
    m = C.shape[0]
    dist = min([hamming_weight(c) for c in C.T])
    for _ in range(50):
        C_mask = C.copy()
        for i in range(m):
            if np.random.rand() > p:
                C_mask[i] = 0

        kernel = C_mask.null_space()
        if len(kernel) > 0:
            for v in kernel:
                v = v.reshape(-1, 1)
                dist = min(dist, hamming_weight(C @ v))

    return dist


def min_distance(C: "galois.FieldArray"):
    """
    Compute the minimum distance of a code generated by C
    """
    C = C.column_space().T
    m, n = C.shape
    dist = m
    for i in range(1, 2**n):
        x = int2bin(i, n).reshape(-1, 1)
        dist = min(dist, hamming_weight(C @ x))

    return dist


# def weight_distribution(C: "galois.FieldArray", times = 2**12, p = None):
#     """
#     Compute the weight distribution of a code generated by C
#     """
#     C = C.column_space().T
#     m, n = C.shape
#     weight_list = []
#     if p is None:
#         for _ in range(times):
#             c = sample_column_space(C)
#             weight_list.append(hamming_weight(c)/m)
#     else:
#         for _ in range(times):
#             C_mask = C.copy()
#             for i in range(m):
#                 if np.random.rand() > p:
#                     C_mask[i] = 0
            
#             kernel = C_mask.null_space()
#             if len(kernel) > 0:
#                 for v in kernel:
#                     v = v.reshape(-1, 1)
#                     weight_list.append(hamming_weight(C @ v)/m)


#     return weight_list


def weight_distribution_worker(args):
    C, start, end, p, m = args
    weight_list = []
    if p is None:
        for _ in range(start, end):
            c = sample_column_space(C)
            weight_list.append(hamming_weight(c)/m)
    else:
        for _ in range(start, end):
            C_mask = C.copy()
            for i in range(m):
                if np.random.rand() > p:
                    C_mask[i] = 0
            
            kernel = C_mask.null_space()
            if len(kernel) > 0:
                for v in kernel:
                    v = v.reshape(-1, 1)
                    weight_list.append(hamming_weight(C @ v)/m)
    return weight_list

def weight_distribution(C: "galois.FieldArray", times = 2**12, p = None):
    """
    Compute the weight distribution of a code generated by C
    """
    C = C.column_space().T
    m, n = C.shape
    weight_list = []

    # Number of processes to create
    num_processes = int(multiprocessing.cpu_count() * 0.75)

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Divide the work among the processes
    ranges = [(i * times // num_processes, (i + 1) * times // num_processes) for i in range(num_processes)]

    # Apply the worker function to each portion of the 'times' range
    results = pool.map(weight_distribution_worker, [(C, start, end, p, m) for start, end in ranges])

    # Combine the results from each worker
    for result in results:
        weight_list.extend(result)

    return weight_list