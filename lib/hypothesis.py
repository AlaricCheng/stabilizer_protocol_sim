import numpy as np
import galois

GF = galois.GF(2)

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
