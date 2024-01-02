import numpy as np
from sys import exit
from numpy.random import default_rng
import galois

from lib.utils import check_element, rank, sample_column_space, hamming_weight, solvesystem, rand_inv_mat, wrap_seed, random_solution
from lib.gen_matrix import sample_D, sample_F

GF = galois.GF(2)


def sample_parameters(n, m, g, seed = None):
    rng = wrap_seed(seed)
    for _ in range(100):
        tmp = [i for i in range(g, m, 2) if i >= 4 and i > g] # m1 = g mod 2, m1 > g
        m1 = tmp[rng.binomial(len(tmp)-1, 0.3)]
        d = rng.binomial(int((m1-g)/2), 0.75) # g + 2*d <= m1
        if g + d <= n and n - g - d <= m - m1 and d > 0:
            break

    ## Routine would silently return bad parameters if no good set was found after 30 iterations (now increased to 100).
    ## This does happen if one samples thousands of times.
    ## Rather bail out than return faulty parameters.
    if not (g + d <= n and n - g - d <= m - m1 and d > 0):
        print("!! Failed to find good parameters!")
        exit()

    return m1, d


def add_row_redundancy(H_s: "galois.FieldArray", s: "galois.FieldArray", m2: int, seed = None, rowAlgorithm = 2):
    """
    Generating R_s so that R_s.s = 0 and the joint row space of H_s and R_s is n
    """
    rng = wrap_seed(seed)
    r = rank(H_s)

    row_space_H_s = H_s.row_space()
    s_null_space = s.reshape((1, -1)).null_space()

    if rowAlgorithm == 1: # Version that was used to create the public secret
        full_basis = row_space_H_s
        for p in s_null_space:
            if not check_element(row_space_H_s.T, p):
                full_basis = np.concatenate((full_basis, p.reshape(1, -1)), axis=0)

    elif rowAlgorithm == 2: # Version linked from 2308.07152v1, correction based on commit 930fc0
        full_basis = row_space_H_s
        for p in s_null_space:
            if not check_element(full_basis.T, p):
                full_basis = np.concatenate((full_basis, p.reshape(1, -1)), axis=0)

    elif rowAlgorithm == 3: # Scramble s^perp before generating row randomness
        s_null_space = rand_inv_mat(s_null_space.shape[0], seed = rng) @ s_null_space
        full_basis = row_space_H_s
        for p in s_null_space:
            if not check_element(full_basis.T, p):
                full_basis = np.concatenate((full_basis, p.reshape(1, -1)), axis=0)
    else:
        print("add_row_redundancy(): Algorithm nr. ", rowAlgorithm, "undefined.")
        exit()
    
    R_s = full_basis[r:] # guarantee that rank(H) = n

    while R_s.shape[0] < m2:
        p = sample_column_space(s_null_space.T, seed = rng)
        if hamming_weight(p) != 0:
            R_s = np.concatenate((R_s, p.T), axis=0)

    return R_s


def initialization(n, m, g, m1=None, d=None, seed = None, rowAlgorithm=2):
    """
    Initialization of the stabilizer construction, where H_s = (F, D, 0) 
    """
    rng = wrap_seed(seed)
    if m1 is None or d is None:
        m1, d = sample_parameters(n, m, g, seed = rng)

    D = sample_D(m1, d, seed = rng)
    zeros = GF.Zeros((m1, n-g-D.shape[1]))
    if g == 0:
        H_s = np.concatenate((D, zeros), axis=1)
    else:
        F = sample_F(m1, g, D, seed = rng)
        H_s = np.concatenate((F, D, zeros), axis=1)
    u = GF.Ones((m1, 1))
    s = random_solution(H_s, u, seed = rng)

    R_s = add_row_redundancy(H_s, s, m-m1, seed = rng, rowAlgorithm=rowAlgorithm)
    H = np.concatenate((H_s, R_s), axis=0)

    ## test analytic estimates

    #B  = R_s[:,g:g+d]
    #C  = R_s[:,g+d:]
    m2=m-m1
    BC = R_s[:,g:]
    G  = H.T@H

    print("d", d)
    print("m2", m2)
    print("m2 - rank(BC) =", m2 - rank(BC))
    print("n - g - m2 =", n - g - m2)
    print("slack in bound=", (n-rank(G))-(n-g-m2))


    return H, s.reshape(-1, 1)


def obfuscation(H: "galois.FieldArray", s: "galois.FieldArray", seed = None):
    """
    H <-- P H Q and s <-- Q^{-1} s, 
    where P is a random permutation matrix 
    and Q is a random invertible matrix
    """
    rng = wrap_seed(seed)
    H = rng.permutation(H).view(GF) # row permutations

    Q = rand_inv_mat(H.shape[1], seed = rng)
    H = H @ Q # column operations
    s = np.linalg.inv(Q) @ s

    return H, s


def stabilizer_construction(n, m, g, m1=None, d=None, seed = None, obfuscate=True, rowAlgorithm=2):
    """
    Generate an IQP matrix H and a secret s, so that the correlation function is 2^{-g/2}
    """
    rng = wrap_seed(seed)
    H, s = initialization(n, m, g, m1, d, seed = rng,rowAlgorithm=rowAlgorithm)

    if obfuscate:
        H, s = obfuscation(H, s, seed = rng)

    return H, s


def quad_res_mod_q(q):
    '''
    Generate the list of quadratic residues modulo q.
    '''
    QRs = []
    for m in range(q):
        QRs.append(m**2 % q)
    QRs.pop(0)
    return list(set(QRs))


def qrc_construction(n, m, q, seed = None):
    """
    construct QRC-IQP instance
    """
    rng = wrap_seed(seed)
    m1 = q
    r = int((q+1)/2) # dim of QRC

    H_s = GF.Zeros((q, r))
    QRs = quad_res_mod_q(q) # the list of quadratic residues
    for i in range(r):
        for qr in QRs:
            H_s[(qr - 1 + i)%q, i] = 1

    if n > r:
        zeros = GF.Zeros((q, n-r))
        H_s = np.concatenate((H_s, zeros), axis=1)

    u = GF.Ones((m1, 1))
    s = random_solution(H_s, u, seed = rng)

    R_s = add_row_redundancy(H_s, s, m-m1, seed = rng)
    # print(H_s.shape, R_s.shape, rank(H_s))
    H = np.concatenate((H_s, R_s), axis=0)

    H, s = obfuscation(H, s, seed = rng)

    return H, s.reshape(-1, 1)


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
