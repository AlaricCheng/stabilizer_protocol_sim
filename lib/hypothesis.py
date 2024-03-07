import numpy as np
import galois
from lib.utils import get_H_s, rank, hamming_weight, get_D_space, check_D_doubly_even

GF = galois.GF(2)

def correlation_function(H: "galois.FieldArray", s: "galois.FieldArray"):
    H_s = get_H_s(H, s)
    g = rank(H_s.T @ H_s)
    m1, n = H_s.shape

    D = get_D_space(H_s)
    if not check_D_doubly_even(D): # if the dual intersection is not doubly-even
        return 0

    res = 0
    for _ in range(5*2**g): # obtain the sign of <Z_s>
        x = GF.Random((n, 1))
        res += np.cos(np.pi * (m1 - 2*hamming_weight(H_s @ x))/4)
    
    # print(res/5/2**g)
    if res > 0:
        return 2**(-g/2)
    else:
        return -2**(-g/2)

def bias(H, s):
    return (1 + correlation_function(H, s))/2

def hypothesis_test(s, X, bias):
    '''
    Hypothesis test by the verifier.
    Args
        s: secret
        X: the set of samples from the prover
        bias: the expected bias
    '''
    if X is None:
        return False
    tolarence = 2/np.sqrt(len(X))
    count = 0
    s = s.flatten()
    for x in X:
        if np.dot(s, x) == 0:
            count += 1
    samp_bias = count/len(X)
    if abs(samp_bias - bias) < tolarence:
        return True
    else: 
        return False
