# %%
import numpy as np
from numpy.random import default_rng
import galois

GF = galois.GF(2)

# %%
def int2bin(n, length):
    '''
    Convert the integer `n` into binary representation of length `length`
    '''
    bin_list = list(bin(n)[2:].zfill(length))
    return GF(bin_list)

def solvesystem(A, b = None, all_sol = False):
    '''
    Solve A x = b over GF(2)
    '''
    assert len(A) != 0, "Empty matrix!"
    A = A.view(GF)
    n = A.shape[1] # n_cols
    null = A.null_space()
    if all_sol == True:
        complete_set = [int2bin(i, len(null)) @ null for i in range(1, 2**(len(null)))] # # linear combination of all vectors in null space

    if (b is None) or np.all(b == 0): # homogeneous equations
        assert np.all(A @ null.transpose() == 0)
        if all_sol == True:
            return GF(complete_set)
        return null
    else: # b != 0
        assert A.shape[1] == b.shape[0], "Inconsistent shapes"
        Ab = np.hstack((A, b.reshape(-1, 1)))
        Ab_reduced = Ab.row_reduce() # Gaussian elimination
        A = Ab_reduced[:, :n]
        b = Ab_reduced[:, n]
        free_var = np.all(A == 0, axis=1).nonzero()[0] # indices for free variables
        if np.all(b[free_var] == 0): 
            if all_sol == True:
                return GF(complete_set) + b
            return b, null
        else: # case: no solution
            return [] 


# %%
if __name__ == "__main__":
    tab = np.loadtxt("../examples/4qubit_rank_1.tab", dtype = int)
    A = tab[:, :4]
    b = GF([1, 1, 1, 1])
    print(solvesystem(A, b, all_sol=False))
