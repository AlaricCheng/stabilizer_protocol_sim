import numpy as np
from lib.construction import stabilizer_construction
from lib.hypothesis import hypothesis_test, correlation_function, bias
from lib.utils import rank, dumpToUuid
from lib.new_attacks import radicalAttack, hammingRazor

def testStabAttack(N=100,n=300,m=360,g=4,m1=None, d=None, d1=None, rowAlgorithm=3,initAlg=2, **kwargs):
    for i in range(N):
        H, secret = stabilizer_construction(n, m, g, m1, d, d1, rowAlgorithm=rowAlgorithm, initAlg=initAlg)
        secret = secret.T[0]

        s,_ = radicalAttack(H,**kwargs)

        if not np.all(s==secret):
            print(f"!! wrong secret.")
            # dumpToUuid([H.tolist(),secret.tolist()])
        else:
            print("Success")

        print("",flush=True)
        
        s = hammingRazor(H)
        print(s)
        print(np.all(s==secret))


def test_hamming_razor():
    H, secret = stabilizer_construction(550, 800, m1 = 150, d = 60, g = 10)
    dumpToUuid([H.tolist(),secret.tolist()])
    secret = secret.T[0]
    s = hammingRazor(H, secret = secret)
    print(np.all(s==secret))


test_hamming_razor()
