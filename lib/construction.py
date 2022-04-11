# %%
import numpy as np
from numpy.random import default_rng
import galois
import stim

GF = galois.GF(2)

# %%
class StabilizerConstruction:
    def __init__(self, tab):
        self.tab = tab.astype(int).view(GF)
        self.n = len(self.tab)
        G = tab[:, 0:self.n]
        assert np.all(G == G.T), "G must be symmetric"
        self.G = G.astype(int).view(GF)

    def get_weight(self):
        '''
        Get weight constraints from the stabilizer tableau.
        '''
        weight_dict = {"00": 0, "01": 1, "10": 2, "11": 3}
        weights = []
        r = self.tab[:, 2*self.n]
        for i in range(self.n):
            weight = f"{self.G[i, i]}" + f"{r[i]}"
            weights.append(weight_dict[weight])
        return np.array(weights)
        
    def init_factor(self):
        '''
        Construct the initial factorization, based on [Lempel 75]. 
        '''
        row_sum = np.sum(self.G, axis = 1)
        N1 = np.nonzero(row_sum)[0]

        E = np.zeros((len(N1), self.n))
        for i in range(len(N1)):
            E[i, N1[i]] = 1
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.G[i, j] == 1:
                    tmp = np.zeros((1, self.n))
                    tmp[0, [i, j]] = 1, 1
                    E = np.append(E, tmp, axis = 0)
        return E.astype(int).view(GF)

    def satisfy_weight_constraint(self, E):
        '''
        Make the factor satisfy the weight constraint.
        '''
        weights = np.sum(E.view(np.ndarray), axis = 0) % 4
        weight_diff = (self.get_weight() - weights) % 4
        for idx, w in enumerate(weight_diff):
            if w == 0:
                continue
            rows = np.zeros((w, self.n))
            rows[:, idx] = np.ones(w)
            E = np.append(E, rows.astype(int).view(GF), axis = 0)
        return E

    def injecting_ones(self):
        pass


# %%
if __name__ == "__main__":
    G = np.loadtxt("../examples/4qubit_rank_1.tab")
    sc = StabilizerConstruction(G)
    E = sc.init_factor()
    print(sc.satisfy_weight_constraint(E))