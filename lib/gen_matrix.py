import numpy as np
from numpy.random import default_rng
import galois

from lib.utils import check_element, rank, fix_basis, sample_column_space, hamming_weight

GF = galois.GF(2)

def random_doubly_even_vector(m1):
    """
    Sample a random vector of length t with weight a multiple of 4
    """
    multiples_of_4 = [i for i in range(4, m1, 4)]
    h = default_rng().choice(multiples_of_4)
    vector = GF.Zeros(m1)
    idx = default_rng().choice(m1, h, replace=False) # idx to be flipped
    vector[idx] = 1

    return vector


def full_rank_symmetric_matrix(g, g_00 = None):
    """
    Sample a random full rank symmetric matrix of size g
    """
    def generate_one(g):
        G = GF.Random((g, g))
        if g_00 is not None:
            G[0, 0] = g_00
        for i in range(g-1):
            for j in range(i+1, g):
                G[j, i] = G[i, j]

        return G
    
    G = generate_one(g)
    while rank(G) < g:
        G = generate_one(g)
    
    return G


def sample_even_parity_vector(basis: "galois.FieldArray"):
    """
    Sample a random vector of even parity from the column space of basis
    """
    for _ in range(20):
        a = sample_column_space(basis)
        if hamming_weight(a) % 2 == 0 and hamming_weight(a) != 0:
            return a
        

def sample_odd_parity_vector(basis: "galois.FieldArray"):
    """
    Sample a random vector of odd parity from the column space of basis
    """
    for _ in range(20):
        a = sample_column_space(basis)
        if hamming_weight(a) % 2 == 1:
            return a


def sample_D(m1, d):
    """
    Sample the generator matrix D of a random doubly-even code of length m1 and dimension d
    """
    c = random_doubly_even_vector(m1)
    D = c.reshape(-1, 1) # initialization

    for _ in range(d-1):
        kernel_D = D.T.null_space().T
        kernel_D = fix_basis(D, kernel_D)
        complement_subspace_basis = kernel_D[:, D.shape[1]:] # ker(D^T) / <c_1, ..., c_t>
        a1 = sample_even_parity_vector(complement_subspace_basis)
        
        if hamming_weight(a1) % 4 == 0:
            D = np.concatenate((D, a1), axis = 1)
        else:
            tmp = np.concatenate((D, a1), axis = 1)
            kernel_D = fix_basis(tmp, kernel_D)
            complement_subspace_basis = kernel_D[:, tmp.shape[1]:] # ker(D^T) / <c_1, ..., c_t, a_1>

            a2 = sample_even_parity_vector(complement_subspace_basis)
            if a2 is None:
                break
            if hamming_weight(a2) % 4 == 0:
                D = np.concatenate((D, a2), axis = 1)
            else:
                D = np.concatenate((D, a1 + a2), axis = 1)
    
    u = GF.Ones((m1, 1))
    if check_element(D, u):
        D = fix_basis(u, D)

    return D


def sample_F(m1, g, D):
    """
    Sample F = (c_1, ..., c_g) so that D^T F = 0 and rank(F^T F) = g
    """
    u = GF.Ones((m1, 1))
    kernel_D = D.T.null_space().T
    kernel_D = fix_basis(D, kernel_D)

    if m1 % 2 == 1:
        F = u
    elif m1 % 2 == 0 and not check_element(D, u):
        c1 = u
        complement_subspace_basis = kernel_D[:, D.shape[1]:]
        c2 = sample_odd_parity_vector(complement_subspace_basis)
        F = np.concatenate((c1 + c2, c2), axis = 1)
    else:
        complement_subspace_basis = kernel_D[:, D.shape[1]:]
        c1 = sample_column_space(complement_subspace_basis)
        C = np.concatenate((D, c1), axis = 1)
        kernel_D = fix_basis(C, kernel_D)
        complement_subspace_basis = kernel_D[:, C.shape[1]:]

        for _ in range(20):
            c2 = sample_column_space(complement_subspace_basis)
            if np.inner(c1.T, c2.T)[0, 0] == 1:
                F = np.concatenate((c1, c2), axis = 1)
                break
    
    return F
        

