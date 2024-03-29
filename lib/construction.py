import numpy as np
from sys import exit
import galois
from pyldpc import make_ldpc

from lib.utils import check_element, rank, sample_column_space, hamming_weight, solvesystem, rand_inv_mat, wrap_seed, random_solution, estimate_distance, sample_sparse_vector
from lib.gen_matrix import sample_D, sample_F, random_doubly_even_vector, concatenated_D, concatenated_code

GF = galois.GF(2)


def sample_parameters(n, m, g, seed = None):
    rng = wrap_seed(seed)
    for _ in range(100):
        tmp = [i for i in range(g, m, 2) if i >= 4 and i > g] # m1 = g mod 2, m1 > g
        m1 = tmp[rng.binomial(len(tmp)-1, 0.3)]
        d = rng.binomial(int((m1-g)/2), 0.75) # g + 2*d <= m1
        if g + d <= n and n - g - d < m - m1 and d > 0:
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


def generate_H_s(n, g, m1, d, seed = None, concat_D = False, m0 = None, d0 = None):
    """
    Initialization of H_s = (F, D, 0) and s
        - concat_D (bool): whether to use code concatenation for D
        - The second layer of codes are of dimension m0 * d0
    """
    rng = wrap_seed(seed)

    if concat_D:
        assert m0 is not None and d0 is not None, "!!m0 and d0 must be specified"
        assert m1 % m0 == 0, "!!m1 % m0 != 0"
        k = m1 // m0
        assert d0 * k >= d, "!!d0 * k < d"
        for _ in range(20):
            K_inner = GF.Random((d0*k, d)) 
            if rank(K_inner) == d:
                break
        D = concatenated_D(m1, d, m0, d0, K_inner)
    else:
        D = sample_D(m1, d, seed = rng)
    
    if g == 0:
        FD = D
    else:
        F = sample_F(m1, g, D, seed = rng)
        FD = np.concatenate((F, D), axis=1)
    
    right_padded = GF.Zeros((m1, n-g-d))
    H_s = np.concatenate((FD, right_padded), axis=1)
    u = GF.Ones((m1, 1))
    s = random_solution(H_s, u, seed = rng)

    return H_s, s


def initialization(n, m, g, m1=None, d=None, seed = None, rowAlgorithm=2, concat_D = False, m0 = None, d0 = None):
    """
    Initialization of the stabilizer construction, where H_s = (F, D, 0) 
    """
    rng = wrap_seed(seed)
    if m1 is None or d is None:
        m1, d = sample_parameters(n, m, g, seed = rng)

    H_s, s = generate_H_s(n, g, m1, d, seed = rng, concat_D = concat_D, m0 = m0, d0 = d0)

    R_s = add_row_redundancy(H_s, s, m-m1, seed = rng, rowAlgorithm=rowAlgorithm)
    H = np.concatenate((H_s, R_s), axis=0)

    ## test analytic estimates
    m2=m-m1
    BC = R_s[:,g:]
    G  = H.T@H

    print("d", d)
    print("m2", m2)
    print("m2 - rank(BC) =", m2 - rank(BC))
    print("n - g - m2 =", n - g - m2)
    print("slack in bound=", (n-rank(G))-(n-g-m2))
    print("relative distance (H_s and R_s):", estimate_distance(H_s)/m1, estimate_distance(R_s)/m2)

    return H, s.reshape(-1, 1)


def initialization_block(n, m, g, m1=None, d=None, d1 = None, seed = None, concat_D = False, m0 = None, d0 = None, AB_type = "zero", concat_C1 = False):
    """
    Initialization of the stabilizer construction, where H_s = (F, D, 0)
    and R_s = (A, B, C_1, C_2), where 
        - C_1 of size (m_2, d_1) generates a doubly-even code
          d1 <= m2 - n + g + d 
        - AB_type = "independent" or "from_C" or "zero"
    """
    rng = wrap_seed(seed)
    if m1 is None or d is None:
        m1, d = sample_parameters(n, m, g, seed = rng)
    
    # generate H_s
    H_s, s = generate_H_s(n, g, m1, d, seed = rng, concat_D = concat_D, m0 = m0, d0 = d0)
    
    # Generate R_s
    m2 = m - m1
    if d1 == 0:
        C = GF.Random((m2, n-g-d), seed = rng)
    else:
        if d1 is None:
            d1 = m2 - n + g + d - 2
        # generate C_1
        if concat_C1:
            assert m2 % m0 == 0, "!!m2 % m0 != 0"
            k = m2 // m0
            assert d0 * k >= d1, "!!d0 * k < d1"
            for _ in range(20):
                K_inner = GF.Random((d0*k, d1), seed = rng) 
                if rank(K_inner) == d1:
                    break
            C = concatenated_D(m2, d1, m0, d0, K_inner)
        else:
            C = sample_D(m2, d1, seed = rng) 
        kernel_C1 = C.T.null_space().T
        # generate C_2
        d2 = n-g-d-d1 #  #columns of C_2
        if (d2 + m2) % 2 == 0:
            C2 = sample_F(m2, d2, C, seed = rng)
        else:
            C2 = sample_F(m2, d2 - 1, C, seed = rng)
        C = np.concatenate((C, C2), axis = 1)
        if C.shape[1] < n-g-d:
            for i in range(20):
                c = sample_column_space(kernel_C1, seed = rng)
                if not check_element(C, c):
                    break
                if i == 19:
                    print("!! cannot form C")
                    exit()
            C = np.concatenate((C, c), axis = 1)
    
    if AB_type == "independent":
        BC = C
        for _ in range(d):
            for i in range(20):
                c = GF.Random((m2, 1), seed = rng)
                if not check_element(BC, c):
                    break
            BC = np.concatenate((BC, c), axis = 1)
        
        R_s = BC
        for _ in range(g):
            c = sample_column_space(BC, seed = rng)
            R_s = np.concatenate((c, R_s), axis = 1)
    elif AB_type == "from_C":         
        R_s = C
        for _ in range(g+d):
            c = sample_column_space(C, seed = rng)
            R_s = np.concatenate((c, R_s), axis = 1)
    elif AB_type == "zero":
        R_s = np.concatenate((GF.Zeros((m2, g+d)), C), axis = 1)
    elif AB_type == "concat":
        # assert m2 % m0 == 0, "!!m2 % m0 != 0"
        # k = m2 // m0
        # assert d0 * k >= g+d, "!!d0 * k < g+d"
        # for _ in range(20):
        #     K_inner = GF.Random((g+d, g+d), seed = rng) 
        #     if rank(K_inner) == g+d:
        #         break
        # K_inner = np.concatenate((K_inner, GF.Zeros((d0*k-g-d, g+d))), axis = 0)
        # AB = concatenated_code(m2, g+d, m0, d0, K_inner)
        # R_s = np.concatenate((AB, C), axis = 1)
        AB = GF.Identity(g+d)
        AB = np.concatenate((AB, GF.Zeros((m2-g-d, g+d))), axis = 0)
        R_s = np.concatenate((AB, C), axis = 1)
    
    # set R_s.s = 0
    supp_s = s.nonzero()[0]
    if hamming_weight(s) == 1:
        R_s[:, supp_s[0]] = 0
    else:
        try:
            j0 = rng.choice(list(set(supp_s) & set(range(g))))
        except:
            j0 = rng.choice(list(set(supp_s) & set(range(g+d)))) # choose a random entry from the first g+d coordinates that are in supp_S
        j0_mask = [j != j0 and j in supp_s for j in range(n)]
        R_s[:, j0] = R_s[:, j0_mask].sum(axis = 1)
    
    # get H
    H = np.concatenate((H_s, R_s), axis=0)

    ## test analytic estimates
    m2=m-m1
    BC = R_s[:, g:]
    C = R_s[:, g+d:]
    G  = H.T@H

    print("d", d)
    print("m2", m2)
    print("d1", d1)
    print("m2 - rank(BC) =", m2 - rank(BC))
    print("n - g - m2 =", n - g - m2)
    print("slack in bound=", (n-rank(G))-(n-g-m2))
    print("relative distance (H_s, R_s, C and H):", estimate_distance(H_s)/m1, estimate_distance(R_s)/m2, estimate_distance(C)/m2, estimate_distance(H)/m)
    print("dim ker(G) = ", n - rank(G))

    return H, s.reshape(-1, 1)


def initialization_sparse(n, m, g, m1=None, d=None, seed = None, m0 = None, d0 = None, C_sparsity = 10, B_sparsity = 10):
    """
    Initialization of the stabilizer construction, where H_s = (F, D, 0)
    and R_s = (A, B, C_1, C_2), where 
        - C_1 of size (m_2, d_1) generates a doubly-even code
          d1 <= m2 - n + g + d 
    """
    rng = wrap_seed(seed)
    if m1 is None or d is None:
        m1, d = sample_parameters(n, m, g, seed = rng)
    
    # generate H_s
    H_s, s = generate_H_s(n, g, m1, d, seed = rng, concat_D = True, m0 = m0, d0 = d0)

    # Generate R_s
    m2 = m - m1
    C = sample_sparse_vector(m2, C_sparsity)
    for _ in range(n-g-d-1):
        c = sample_sparse_vector(m2, C_sparsity, C)
        C = np.concatenate((C, c), axis = 1)
    
    BC = C
    for _ in range(d):
        c = sample_sparse_vector(m2, B_sparsity, BC)
        BC = np.concatenate((c, BC), axis = 1)

    R_s = np.concatenate((GF.Random((m2, g)), BC), axis = 1)
    # set R_s.s = 0
    supp_s = s.nonzero()[0]
    if hamming_weight(s) == 1:
        R_s[:, supp_s[0]] = 0
    else:
        try:
            j0 = rng.choice(list(set(supp_s) & set(range(g))))
        except:
            j0 = rng.choice(list(set(supp_s) & set(range(g+d)))) # choose a random entry from the first g+d coordinates that are in supp_S
        j0_mask = [j != j0 and j in supp_s for j in range(n)]
        R_s[:, j0] = R_s[:, j0_mask].sum(axis = 1)

    # get H
    H = np.concatenate((H_s, R_s), axis=0)

    ## test analytic estimates
    m2=m-m1
    G  = H.T@H

    print("d", d)
    print("m2", m2)
    print("m2 - rank(BC) =", m2 - rank(BC))
    print("n - g - m2 =", n - g - m2)
    print("slack in bound=", (n-rank(G))-(n-g-m2))
    print("relative distance (H_s, R_s, C and H):", estimate_distance(H_s)/m1, estimate_distance(R_s)/m2, estimate_distance(C)/m2, estimate_distance(H)/m)
    print("dim ker(G) = ", n - rank(G))

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


def stabilizer_construction(n, m, g, m1=None, d=None, d1=None, seed = None, obfuscate=True, rowAlgorithm=2, initAlg = 2, concat_D = False, m0 = None, d0 = None, AB_type = "zero", concat_C1 = False, C_sparsity = 10, B_sparsity = 10):
    """
    Generate an IQP matrix H and a secret s, so that the correlation function is 2^{-g/2} 
    """
    rng = wrap_seed(seed)
    if initAlg == 1:
        H, s = initialization(n, m, g, m1, d, seed = rng,rowAlgorithm=rowAlgorithm, concat_D = concat_D, m0 = m0, d0 = d0)
    elif initAlg == 2:
        H, s = initialization_block(n, m, g, m1, d, d1=d1, seed = rng, concat_D = concat_D, AB_type = AB_type, m0 = m0, d0 = d0, concat_C1 = concat_C1)
    elif initAlg == 3:
        H, s = initialization_sparse(n, m, g, m1, d, seed = rng, m0 = m0, d0 = d0, C_sparsity = C_sparsity, B_sparsity = B_sparsity)

    if obfuscate:
        H, s = obfuscation(H, s, seed = rng)

    print("rank(H) =", rank(H))

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
