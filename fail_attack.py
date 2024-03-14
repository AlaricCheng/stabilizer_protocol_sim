import numpy as np
import galois
from lib.construction import stabilizer_construction
from lib.hypothesis import hypothesis_test, correlation_function, bias
from lib.utils import rank, dumpToUuid, estimate_distance, solvesystem, get_H_s, get_R_s, hamming_weight
from lib.gen_matrix import sample_D, sample_F
from lib.new_attacks import radicalAttack, hammingRazor, lazyLinearityAttack, doubleMeyer
import json
import argparse

GF = galois.GF(2)

def random_vs_H_s():
    g = 10
    res = []
    for m in range(300, 600, 20):
        for _ in range(5):
            d = int((m - g)/2) - 3
            D = sample_D(m, d)
            F = sample_F(m, g, D)
            H = np.concatenate((D, F), axis=1)
            H_rand = GF.Random((m, g + d))
            print("m, d, g:", m, d, g, "delta(H_s):", estimate_distance(H)/m, "delta(H_rand):", estimate_distance(H_rand)/m)
            res.append([m, d, g, estimate_distance(H), estimate_distance(H_rand)])

    dumpToUuid(res)

        

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


def hamming_razor_iter():
    g, m1, d = 10, 150, 60
    for i in range(20):
        m = 400 + i*20
        n = int(m/2 + 100)
        d1 = m-m1-n+g+d-3
        print(n, m, g, m1, d, d1)

        H, secret = stabilizer_construction(n, m, g, m1, d, d1, initAlg=2)
        secret = secret.T[0]
        s = hammingRazor(H, secret = secret)
        print(np.all(s==secret))


def distance_compare():
    mgd_pair = ((100, 2, 47), (100, 10, 43), (150, 6, 70), (200, 10, 93))
    res = {}
    for pair in mgd_pair:
        m1, g, d = pair
        res[pair] = []
        for _ in range(20):
            D = sample_D(m1, d)
            F = sample_F(m1, g, D)
            H_s = np.concatenate((D, F), axis=1)

            H_s_dist = estimate_distance(H_s, times = 1000)
            random_distance = estimate_distance(GF.Random((m1, g+d)), times = 1000)

            res[pair].append([H_s_dist, random_distance])
            print(pair, H_s_dist, random_distance)

    dumpToUuid(res)


def test_H_not_full_rank():
    H, secret = stabilizer_construction(50, 200, 1, initAlg=1)
    s, _ = lazyLinearityAttack(H, g_thres = 2)
    if np.all(s == secret):
        print("Success")
    H_extended = np.concatenate((H, GF.Zeros((200, 50))), axis=1)
    # H_reduced = H_extended.column_space().T
    print("shape and rank of H_reduced", H_extended.shape, rank(H_extended))
    s, _ = lazyLinearityAttack(H_extended, g_thres = 2)
    if s is None:
        print("Failed")
    else:
        print((H_extended @ s == H @ secret).all())


def hamming_razor_fine_tune():
    with open(args.read, "r") as f:
        print("reading from", args.read)
        H, secret = json.load(f)
        H = GF(H)
        secret = GF(secret)

    s, support = hammingRazor(H, secret = secret, return_support=True, endurance=args.E, p = args.p)
    if np.all(s==secret.T[0]):
        print("hamming's razor succeeded")

    R_s_candidate = H[support == 1]
    print("good support?", (R_s_candidate @ secret == 0).all() and len(R_s_candidate) != 0)

    # Hamming's razor + lazy linearity attack
    H_masked = H.copy()
    H_masked[support == 1] = 0
    H_reduced = H_masked.column_space().T
    print("reduced H rank", rank(H_reduced), "shape", H_reduced.shape)

    s, _ = lazyLinearityAttack(H_reduced, g_thres=10, endurance=50)
    if s is None:
        print("Failed")
    else:
        print((H_reduced @ s == H @ secret).all())

    # hamming's razor + double Meyer
    s, _ = doubleMeyer(H_reduced, g_thres = 10, kfold = 3, endurance = 20)
    if s is None:
        print("Failed")
    else:
        print((H_reduced @ s == H @ secret).all())


def test_hamming_razor(n=300, m=500, g=10, m1=250, d=115, d1=84):
    if args.read:
        with open(args.read, "r") as f:
            H, secret = json.load(f)
        H = GF(H)
        secret = GF(secret)
    else:
        initAlg = 3 if args.C_sparsity else 2
        H, secret = stabilizer_construction(n, m, g, m1, d, d1, initAlg=initAlg, obfuscate = False, AB_type = args.AB_type, concat_D = args.concat_D, concat_C1 = args.concat_C1, m0 = args.m0, d0 = args.d0, C_sparsity = args.C_sparsity, B_sparsity = args.B_sparsity)
        if args.dump:
            dumpToUuid([H.tolist(),secret.tolist()])

    # first try out radical attack
    s,_ = radicalAttack(H)
    print("radical attack succeeded?", np.all(s==secret))

    # linearity attack
    # s, _ = lazyLinearityAttack(H, g_thres=10, endurance=1000)

    # then try out hamming razor
    for p in np.arange(0.05, 0.4, 0.05):
        s, support = hammingRazor(H, secret = secret, return_support=True, endurance=500, p = p)
        if np.all(s==secret.T[0]):
            print("hamming's razor succeeded")
            exit()

        R_s_candidate = H[support == 1]
        good_support = (R_s_candidate @ secret == 0).all() and len(R_s_candidate) != 0
        print("good support?", good_support)
        # if not good_support:
        #     exit()

        # Hamming's razor + lazy linearity attack
        H_masked = H.copy()
        H_masked[support == 1] = 0
        H_reduced = H_masked.column_space().T
        print("reduced H rank", rank(H_reduced), "shape", H_reduced.shape)

        s, _ = lazyLinearityAttack(H_reduced, g_thres=10, endurance=50)
        if s is None:
            print("Failed")
        else:
            print((H_reduced @ s == H @ secret).all())

        # hamming's razor + double Meyer
        s, _ = doubleMeyer(H_reduced, g_thres = 10, kfold = 3, endurance = 20)
        if s is None:
            print("Failed")
        else:
            print((H_reduced @ s == H @ secret).all())

    # radical attack + row deletion
    # for _ in range(100):
    #     idx = np.random.choice(m, 5, replace=False)
    #     H_reduced = np.delete(H, idx, axis=0)
    #     s, _ = radicalAttack(H_reduced)
    #     if np.all(s==secret):
    #         print("radical attack + row deletion succeeded")
    #         exit()
    # print("radical attack + row deletion failed")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help = "number of columns in H")
    parser.add_argument("-m", type=int, help = "number of rows in H")
    parser.add_argument("-g", type=int, help = "g value for the correlation function")
    parser.add_argument("-m1", type=int, help = "number of rows in H_s")
    parser.add_argument("-d", type=int, help = "number of columns in D")
    parser.add_argument("-d1", type=int, help = "number of columns in C_1")
    parser.add_argument("--AB-type", type=str, default="zero")
    parser.add_argument("--concat_D", action="store_true", help = "whether to use code concatenation for D")
    parser.add_argument("--concat_C1", action="store_true", help = "whether to use code concatenation for C_1")
    parser.add_argument("--C_sparsity", type = int, help = "sparsity of C")
    parser.add_argument("--B_sparsity", type = int, help = "sparsity of B")
    parser.add_argument("-m0", type = int, help = "length of the smaller code in the concatenated code")
    parser.add_argument("-d0", type = int, help = "dimension of the smaller code in the concatenated code")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--read", type = str)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--E", type=int, default=500)
    parser.add_argument("--p", type=float, default=0.3)

    args = parser.parse_args()

    if args.fine_tune:
        hamming_razor_fine_tune()
        exit()

    test_hamming_razor(n=args.n, m=args.m, g=args.g, m1=args.m1, d=args.d, d1=args.d1)






# make the parameters big
# similar_m1_and_m2(n=1128, m=2000, g=10, m1=1000, d=493, d1=373)
# similar_m1_and_m2(n=1000, m=2000, g=10, m1=1000, d=493, d1=493)
# similar_m1_and_m2(n=1020, m=2000, g=10, m1=1000, d=493, d1=483)
# similar_m1_and_m2(n=1040, m=2000, g=10, m1=1000, d=493, d1=463)
# similar_m1_and_m2(n=1060, m=2000, g=10, m1=1000, d=493, d1=443)
# similar_m1_and_m2(n=1080, m=2000, g=10, m1=1000, d=493, d1=423)
# similar_m1_and_m2(n=1100, m=2000, g=10, m1=1000, d=493, d1=403)
# similar_m1_and_m2(n=2128, m=4000, g=10, m1=2000, d=993, d1=875)
# similar_m1_and_m2(n=750, m=1200, g=10, m1=300, d=143, d1=1)


# random_vs_H_s()