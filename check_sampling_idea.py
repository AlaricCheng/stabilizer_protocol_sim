from lib import iqp_sim
import numpy as np
import galois
from lib.utils import rank, int2bin
from numpy.linalg import matrix_rank, solve, lstsq
# from scipy.linalg import solve

GF = galois.GF(2)

H = np.loadtxt("./temp/H_10_qubit", dtype=int, delimiter=" ").view(GF)
prob = np.loadtxt("./temp/prob_10_qubit", dtype=float)
# print(H.shape, prob.shape)
# s = GF.Random(10)

# print(iqp_sim.correlation_function_from_dist(prob, s))

# print(iqp_sim.correlation_function_from_H(H, s))

S = []

while len(S) < 20:
    s = GF.Random(10)
    if iqp_sim.correlation_function_from_H(H, s) >= 2**(-3/2):
        S.append(s)

S.append(GF.Zeros(10))

cor_list = np.array([iqp_sim.correlation_function_from_dist(prob, s) for s in S])

S = GF(S)
S = np.unique(S, axis=0).view(GF)
cor_list = cor_list[np.unique(S, axis = 0, return_index = True)[1]]

f = lambda a: 1 - 2*int(a)

ind_set = []
idx_set = []
for i in range(2**10):
    x = int2bin(i, 10).reshape(-1, 1)
    b = np.array([f(a) for a in (S @ x)]).reshape(-1, 1)
    ind_set.append(b)
    if len(ind_set) <= len(S):
        if len(ind_set) > matrix_rank(np.hstack(ind_set)):
            ind_set.pop()
        else:
            idx_set.append(x)
    else:
        idx_set.append(x)
    if len(ind_set) == 5*len(S):
        break

ind_set = np.hstack(ind_set)
print(ind_set.shape, len(idx_set), i)

q = lstsq(ind_set, cor_list, rcond = None)[0]

print(q)

print(np.all(q >= 0))

print(np.allclose(ind_set @ q, cor_list))



