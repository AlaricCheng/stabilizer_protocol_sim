from typing import List, Union
import numpy as np
import galois
import qutip as qt
from lib.utils import int2bin, rank

GF = galois.GF(2)

def iqp_output_prob_dist(
    H: 'galois.GF2', 
    theta: Union[float, List[float]]
) -> 'np.ndarray':
    '''
    Given an IQP matrix and the angles, return the probability distribution of the output state.
    '''
    H = H.view(np.ndarray)
    if isinstance(theta, float):
        theta = [theta] * H.shape[0]
    n = H.shape[1] # number of qubits
    
    state = qt.basis([2]*n, [0]*n)
    for i, p in enumerate(H):
        exponent = (1j) * theta[i] * qt.tensor([qt.sigmax()**(ele) for ele in p])
        state = exponent.expm() * state
    
    return np.abs(state[:, 0].flatten())**2


def correlation_function_from_dist(
    prob: 'np.ndarray', 
    s: 'galois.GF2'
):
    '''
    Given the probability distribution and the secret, return the correlation function.
    '''
    s = s.view(np.ndarray)
    res = 0
    for i in range(len(prob)):
        x = int2bin(i, len(s)).view(np.ndarray)
        res += prob[i] * (-1)**(np.dot(x, s))
    
    return res


def correlation_function_from_H(
    H: 'galois.GF2', 
    s: 'galois.GF2'
):
    '''
    Given the IQP matrix, and the secret, return the correlation function. It actually return 2^(-g/2), where g is rank(H_s^T H_s)
    '''
    idx = H @ s.reshape(-1, 1)
    H_s = H[(idx == 1).flatten()]
    g = rank(H_s.T @ H_s)

    return 2**(-g/2)