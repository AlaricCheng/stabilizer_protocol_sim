import numpy as np
from numpy.random import default_rng
import galois
from scipy.linalg import block_diag
from sys import exit
import multiprocessing

from lib.utils import check_element, rank, fix_basis, sample_column_space, hamming_weight, wrap_seed

GF = galois.GF(2)

def random_doubly_even_vector(m1, seed = None, low_weight = True):
    """
    Sample a random vector of length t with weight a multiple of 4
    """
    rng = wrap_seed(seed)
    multiples_of_4 = [i for i in range(4, m1+1, 4)]
    if m1 > 16:
        h = rng.choice(multiples_of_4)#[:4])
    else:
        h = 4
    vector = GF.Zeros(m1)
    idx = rng.choice(m1, h, replace=False) # idx to be flipped
    vector[idx] = 1

    return vector


# def sample_even_parity_vector(basis: "galois.FieldArray", seed = None):
#     """
#     Sample a random vector of even parity from the column space of basis
#     """
#     rng = wrap_seed(seed)
#     for _ in range(20):
#         a = sample_column_space(basis, seed = rng)
#         if hamming_weight(a) % 2 == 0 and hamming_weight(a) != 0:
#             return a

def sample_even_parity_vector_worker(args):
    basis, seed = args
    assert type(seed) == int
    a = sample_column_space(basis, seed = seed)
    if hamming_weight(a) % 2 == 0 and hamming_weight(a) != 0:
        return a


def sample_even_parity_vector(basis: "galois.FieldArray", seed = None):
    """
    Sample a random vector of even parity from the column space of basis
    """
    rng = wrap_seed(seed)
    num_processes = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=num_processes)
    seeds = rng.choice(2**32, 20, replace = False)
    results = pool.map(sample_even_parity_vector_worker, [(basis, int(seed)) for seed in seeds])

    for result in results:
        if result is not None:
            return result


def sample_odd_parity_vector(basis: "galois.FieldArray", seed = None):
    """
    Sample a random vector of odd parity from the column space of basis
    """
    rng = wrap_seed(seed)
    for _ in range(20):
        a = sample_column_space(basis, seed = rng)
        if hamming_weight(a) % 2 == 1:
            return a
        

def sample_with_orthogonality_constraint(basis: "galois.FieldArray", a: "galois.FieldArray", seed = None):
    """
    Sample a random vector from the column space of basis that has inner product 1 with a
    """
    rng = wrap_seed(seed)
    for _ in range(20):
        b = sample_column_space(basis, seed = rng)
        if np.inner(a.T, b.T)[0, 0] == 1:
            return b
        

def complement_subspace_basis(V: "galois.FieldArray", W: "galois.FieldArray"):
    """
    return a basis for V / W
    """
    V = fix_basis(W, V)

    return V[:, W.shape[1]:]


def sample_D(m1, d, seed = None):
    """
    Sample the generator matrix D of a random doubly-even code of length m1 and dimension d
    """
    rng = wrap_seed(seed)
    c = random_doubly_even_vector(m1, seed = rng)
    D = c.reshape(-1, 1) # initialization

    for _ in range(d-1):
        kernel_D = D.T.null_space().T

        if rank(kernel_D) == rank(D):
            break

        for i in range(20):
            a1 = sample_even_parity_vector(kernel_D, seed = rng)
            if not check_element(D, a1):
                break
            if i == 19:
                print("!! d may be too large")
                exit()
        
        if hamming_weight(a1) % 4 == 0:
            D = np.concatenate((D, a1), axis = 1)
        else:
            tmp = np.concatenate((D, a1), axis = 1)
            tmp_kernel = tmp.T.null_space().T
            if rank(tmp_kernel) == rank(tmp):
                break
            else:
                for i in range(20):
                    a2 = sample_even_parity_vector(tmp_kernel, seed = rng)
                    if not check_element(tmp, a2):
                        break
                    if i == 19:
                        print("!! d may be too large")
                        exit()
            
            if hamming_weight(a2) % 4 == 0:
                D = np.concatenate((D, a2), axis = 1)
            else:
                D = np.concatenate((D, a1 + a2), axis = 1)

    if not rank(D) == d:
        print("!! rank(D) < d")
        exit()

    return D


def sample_F(m1, g, D, seed = None):
    """
    Sample F = (c_1, ..., c_g) so that D^T F = 0 and rank(F^T F) = g
    """
    rng = wrap_seed(seed)
    u = GF.Ones((m1, 1))
    kernel_D = D.T.null_space().T

    # initialization
    if m1 % 2 == 1:
        F = u
    elif m1 % 2 == 0 and not check_element(D, u):
        for i in range(20):
            c2 = sample_odd_parity_vector(kernel_D, seed = rng)
            if not check_element(D, c2):
                break
            if i == 19:
                print("!! initialization failed")
                exit()

        F = np.concatenate((u + c2, c2), axis = 1)
    else:
        for i in range(20):
            c1 = sample_column_space(kernel_D, seed = rng)
            c2 = sample_column_space(kernel_D, seed = rng)
            if not check_element(D, c1) and not check_element(D, c2) and np.inner(c1.T, c2.T)[0, 0] == 1:
                break
            if i == 19:
                print("!! initialization failed")
                exit()

        F = np.concatenate((c1, c2), axis = 1)
    
    # other columns
    while F.shape[1] < g:
        C = np.concatenate((D, F), axis = 1)  # \calC
        kernel_C = C.T.null_space().T  # \calC^{\perp}
        for i in range(20):
            c1 = sample_column_space(kernel_C, seed = rng)
            c2 = sample_column_space(kernel_C, seed = rng)
            if not check_element(D, c1) and not check_element(D, c2) and np.inner(c1.T, c2.T)[0, 0] == 1:
                break
            if i == 19:
                print("!! increase iterations")
                exit()

        F = np.concatenate((F, c1, c2), axis = 1)
    
    return F
        

def concatenated_D(m1, d, m0, d0, K_inner):
    k = m1 // m0
    assert K_inner.shape == (d0 * k, d), "The inner generator matrix must have shape (d0 * k, d)"

    encoding_list = []
    for _ in range(k):
        encoding_list.append(sample_D(m0, d0))

    encoding_matrix = GF(block_diag(*encoding_list)) 

    D = encoding_matrix @ K_inner

    return D
