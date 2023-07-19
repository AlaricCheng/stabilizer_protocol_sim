import numpy as np
from numpy.random import default_rng
import galois

from lib.utils import check_element, rank, fix_basis, sample_column_space, hamming_weight

GF = galois.GF(2)

def random_doubly_even_vector(m1):
    """
    Sample a random vector of length t with weight a multiple of 4
    """
    multiples_of_4 = [i for i in range(4, m1+1, 4)]
    h = default_rng().choice(multiples_of_4)
    vector = GF.Zeros(m1)
    idx = default_rng().choice(m1, h, replace=False) # idx to be flipped
    vector[idx] = 1

    return vector


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
        

def sample_with_orthogonality_constraint(basis: "galois.FieldArray", a: "galois.FieldArray"):
    """
    Sample a random vector from the column space of basis that has inner product 1 with a
    """
    for _ in range(20):
        b = sample_column_space(basis)
        if np.inner(a.T, b.T)[0, 0] == 1:
            return b
        

def complement_subspace_basis(V: "galois.FieldArray", W: "galois.FieldArray"):
    """
    return a basis for V / W
    """
    V = fix_basis(W, V)

    return V[:, W.shape[1]:]


def sample_D(m1, d):
    """
    Sample the generator matrix D of a random doubly-even code of length m1 and dimension d
    """
    c = random_doubly_even_vector(m1)
    D = c.reshape(-1, 1) # initialization

    for _ in range(d-1):
        kernel_D = D.T.null_space().T
        kernel_D_complement = complement_subspace_basis(kernel_D, D) # ker(D^T) / <c_1, ..., c_t>
        a1 = sample_even_parity_vector(kernel_D_complement)
        
        if hamming_weight(a1) % 4 == 0:
            D = np.concatenate((D, a1), axis = 1)
        else:
            tmp = np.concatenate((D, a1), axis = 1)
            tmp_kernel = tmp.T.null_space().T
            tmp_kernel_complement = complement_subspace_basis(tmp_kernel, tmp) # ker(D^T) / <c_1, ..., c_t, a_1> so that a2 . a1 = 0
            if tmp_kernel_complement.shape[1] != 0: # if tmp_kernel_complement is not empty
                a2 = sample_even_parity_vector(tmp_kernel_complement)
                if a2 is None:
                    break
            else:
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

    # initialization
    if m1 % 2 == 1:
        F = u
    elif m1 % 2 == 0 and not check_element(D, u):
        kernel_D_slash_D = complement_subspace_basis(kernel_D, D)
        c2 = sample_odd_parity_vector(kernel_D_slash_D)

        F = np.concatenate((u + c2, c2), axis = 1)
    else:
        kernel_D_slash_D = complement_subspace_basis(kernel_D, D)
        c1 = sample_column_space(kernel_D_slash_D)
        c2 = sample_with_orthogonality_constraint(kernel_D_slash_D, c1)

        F = np.concatenate((c1, c2), axis = 1)
    
    # other columns
    while F.shape[1] < g:
        C = np.concatenate((D, F), axis = 1)  # \calC
        kernel_C = C.T.null_space().T  # \calC^{\perp}
        kernel_C_slash_D = complement_subspace_basis(kernel_C, D)  # \calC^{\perp} / \calD
        c1 = sample_even_parity_vector(kernel_C_slash_D)
        c2 = sample_with_orthogonality_constraint(kernel_C_slash_D, c1)

        F = np.concatenate((F, c1, c2), axis = 1)
    
    return F
        

