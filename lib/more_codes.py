import numpy as np
import galois
from lib.utils import check_element, wrap_seed

GF2 = galois.GF(2)


def binary_reed_solomon(n, k):
    """
    Return the generator matrix of a binary Reed-Solomon code with parameters (n, k), with n = 2^l - 1.
    """
    rs = galois.ReedSolomon(n, k)
    print(rs)
    GF = rs.field

    G = rs.G
    G = GF2([np.concatenate(G[i].vector()) for i in range(G.shape[0])]).T # GF(2^l) -> GF(2)

    for i in range(rs.k * int(np.log2(n+1)) - k):
        for _ in range(10):
            m = GF.Random(rs.k)
            c = rs.encode(m)
            c = np.concatenate(c.vector()).reshape(-1, 1) # GF(2^l) -> GF(2)
            if not check_element(G, c):
                G = np.concatenate((G, c), axis = 1)
                break

    return G


def shorten_code(G, m, seed = None):
    rng = wrap_seed(seed)
    idx = rng.choice(G.shape[0], m, replace = False)
    G_new = G[idx].column_space().T

    return G_new



if __name__ == "__main__":
    n = 2**6 - 1
    k = 13
    G = binary_reed_solomon(n, k)
    print(G.shape)

    G_new = shorten_code(G, 250)
    print(G_new.shape)
    print(G_new.column_space().T.shape)

