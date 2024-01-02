import numpy as np
import galois
from lib.utils import solvesystem, rank, wrap_seed, get_H_s, iter_column_space, hamming_weight

GF = galois.GF(2)

def radicalAttack(
        H: "galois.FieldArray",
        verbose: bool = True
    ):

    if verbose: 
        print("Attack commencing...")
    n = H.shape[1]
    G = H.T @ H
    toKer = G.null_space().T

    dimKer = n-rank(G)
    if dimKer == 0:
        if verbose: 
            print("!! dim ker G = 0, aborting")
        return (False,False)
    if verbose: 
        print("dim ker G =", dimKer)

    hammings = [sum(v.tolist()) for v in (H@toKer).T]
    if  not all([x%4==0 for x in hammings]):
        # This is a minimalist hack.
        # One should actually construct a basis of a maximal doubly-even subspace.
        # But it does happen to work on the challenge data set 
        # (though this might depend on the specific implementation of null_space())
        if verbose: 
            print("!! Some generators aren't doubly-even. Will throw them out.")
        toKer = ((toKer.T)[[h % 4==0 for h in hammings]]).T

        if toKer.size == 0:
            if verbose: 
                print("!! nothing left after throwing out singly-even generators")
            return (False,False)
    
    coordinateOccupation = sum(np.array((H@toKer).T))
    S = GF([1 if x>0 else 0 for x in coordinateOccupation])

    sol = solvesystem(H,S)
    if (len(sol)==0):
        if verbose: 
            print("!! Can't realize support pattern linearly")
        s=False
    else:
        s=sol[0]

    return (s,S)



def doubleMeyer(
        H: "galois.FieldArray", 
        g_thres: int = 5, 
        endurance: int = 100, 
        ambition: int = 5, 
        kfold: int = 1,
        seed = None,
        verbose = True,
    ):
    rng = wrap_seed(seed)
    count = 0

    while count < endurance:
        
        G_all = GF(np.zeros((1,H.shape[1]),dtype=int))

        for i in range(kfold):
            
            d = GF.Random(H.shape[1], seed = rng)
            H_d = get_H_s(H, d)
            G_d = H_d.T @ H_d
            G_all = np.concatenate((G_all,G_d))

        ker_Gall = G_all.null_space()
        if verbose:
            print("Dimension of ker(G_all): ", len(ker_Gall))

        ker_Gall_space = iter_column_space(ker_Gall.T)

        if len(ker_Gall) > ambition:
            if verbose:
                print('Kernel too big, unambitiously skipping it')
        else:
            for s_i in ker_Gall_space:
                if property_check(H, s_i, g_thres):
                    if verbose:
                        print('Found a secret')
                    return s_i, count
    
        count += 1
    if verbose:
        print('Ran out of steam')
    return None, endurance

def lazyLinearityAttack(
        H: "galois.FieldArray", 
        g_thres: int = 5, 
        endurance: int = 100, 
        ambition: int = 5, 
        seed = None,
        verbose = True
    ):
    rng = wrap_seed(seed)
    count = 0

    while count < endurance:
        d = GF.Random(H.shape[1], seed = rng)
        H_d = get_H_s(H, d)
        G_d = H_d.T @ H_d
        ker_Gd = G_d.null_space()
        if verbose:
            print("Dimension of ker(G_d): ", len(ker_Gd))
        ker_Gd_space = iter_column_space(ker_Gd.T)

        if len(ker_Gd) > ambition:
            if verbose:
                print('Kernel too big, unambitiously skipping it')
        else:
            for s_i in ker_Gd_space:
                check_res = property_check(H, s_i, g_thres)
                if check_res:
                    if verbose:
                        print('Found a secret')
                    return s_i, count
        
        count += 1

    if verbose:
        print('Ran out of steam')
    return None, endurance


def hammingRazor(
        H: "galois.FieldArray", 
        p: float = 0.25,
        endurance: int = 100,
        verbose = True):

    # Runs Hamming's Razor Attack 
    # Input: 
    #   H -- m x n GF2 array
    #   p -- float in [0,1]
    #   E -- int > 0
    # Returns:
    #   s -- n x 1 GF2 array if successful
    #        "False" if not successful

    m = H.shape[0]
    support = np.zeros(m)

    for _ in range(endurance):
        Sc = np.random.rand(m) > p
        HSc = H[Sc,:]

        K = HSc.null_space().T

        support = compress(support + np.array(H@K)@np.ones(K.shape[1]))
    if verbose:
        print(f"Found {sum(support)} candidates for redundant rows")

    sol = solvesystem(H,GF(np.ones(m,int)-support))
    if (len(sol)==0):
        if verbose:
            print("!! Can't realize support pattern linearly")
        s=False
    else:
        s=sol[0]

    return (s)


def property_check(H, s_i, rank_thres=5):
    """
    check whether rank(H_{s_i}^T H_{s_i}) <= rank_thres, and whether D_{s_i} is doubly even
    """
    H_si = get_H_s(H, s_i)
    g = rank(H_si.T @ H_si)

    if g <= rank_thres:
        D = get_D_space(H_si)
        if rank(D) ==0:
            return False
        return check_D_doubly_even(D)
        
    return False


def compress(a):
    # return array with 1's on support of a
    return((a>0).astype(int))

def get_D_space(H: "galois.FieldArray"):
    """
    Get D = C \bigcap C^{\perp} 
    """
    G = H.T @ H
    ker = G.null_space()
    D = H @ ker.T
    return D.column_space().T

def check_D_doubly_even(D):
    """
    check whether D spans a doubly-even code
    """
    
    if D is not None:
        for c in D.T:
            if hamming_weight(c) % 4 != 0:
                return False
    return True

def get_H_s_with_zeros(H: "galois.FieldArray", s: 'galois.FieldArray'):
        '''
        Get H_s by setting rows that are orthogonal to s to zero
        '''
        idx = H @ s.reshape(-1, 1)
        H_s = copy.deepcopy(H)
        H_s[(idx != 1).flatten()] = 0

        return H_s