import numpy as np
import galois
from matplotlib import pyplot as plt
import json, uuid
import argparse
from pyldpc import make_ldpc

from lib.utils import weight_distribution, get_H_s, get_R_s, rank, dumpToUuid, estimate_distance, get_D_space
from lib.gen_matrix import sample_D, sample_F
from lib.more_codes import shorten_code
from lib.construction import generate_H_s

GF = galois.GF(2)


def load_data(uuid):
    with open(f"log/{uuid}.json", "r") as f:
        H, secret = json.load(f)
    H = GF(H)
    secret = GF(secret)

    H_s = get_H_s(H, secret)
    R_s = get_R_s(H, secret)

    return H_s, R_s


def compare_various_codes():
    m1, g, d, m2, r2 = args.m1, args.g, args.d, args.m2, args.r2

    C = GF.Random((m2, r2))
    print("shape and rank of random C:", C.shape, rank(C))
    C_weight = weight_distribution(C, times = args.times, p = args.p)

    BC = GF.Random((m2, d+r2))
    print("shape and rank of random BC:", BC.shape, rank(BC))
    BC_weight = weight_distribution(BC, times = args.times, p = args.p)

    uuid_str = str(uuid.uuid4())
    print(uuid_str)
    
    # if args.read:
    #     with open(f"log/{args.read}.json", "r") as f:
    #         FD = json.load(f)
    #     FD = GF(FD)
    #     uuid_str = args.read
    # else:
    #     D = sample_D(m1, d)
    #     F = sample_F(m1, g, D)
    #     FD = np.concatenate([F, D], axis = 1)
    #     assert rank(FD.T @ FD) == g
    #     with open(f"log/{uuid_str}.json", "w") as f:
    #         json.dump(FD.tolist(), f)
    # FD_weight = weight_distribution(FD, times = args.times, p = args.p)
    # print("FD_weight[:10]", FD_weight[:10])
    # print("FD distance", estimate_distance(FD))

    FD, _ = generate_H_s(g+d, g, m1, d, low_weight = True)

    # D1 = sample_D(2*(d-3)+g, d)
    # D1 = np.concatenate([D1, GF.Zeros((m1 - 2*(d-3)-g, d))], axis = 0)
    # F1 = sample_F(m1, g, D1)
    # FD1 = np.concatenate([F1, D1], axis = 1)
    # assert rank(FD1.T @ FD1) == g
    FD_weight = weight_distribution(FD, times = args.times, p = args.p)
    print("bad FD weight[:10]", FD_weight[:10])
    print("bad FD distance", estimate_distance(FD))

    
    fig, ax = plt.subplots(ncols=2, figsize = (10, 4))
    # ax[0].hist(FD_weight, bins = np.arange(0, 0.3, 0.002), alpha = 0.7, cumulative=True, density = True, label="good FD")
    ax[0].hist(FD_weight, bins = np.arange(0, 0.3, 0.002), alpha = 0.7, cumulative=True, density = True, label="FD")
    ax[0].hist(C_weight, bins = np.arange(0, 0.3, 0.002), alpha = 0.7, cumulative=True, density = True, label="Random C")
    ax[0].hist(BC_weight, bins = np.arange(0, 0.3, 0.002), alpha = 0.7, cumulative=True, density = True, label="Random BC")

    # ax[1].hist(FD_weight, bins = np.arange(0, 0.3, 0.002), alpha = 0.7, cumulative=False, density = True, label="good FD")
    ax[1].hist(FD_weight, bins = np.arange(0, 0.3, 0.002), alpha = 0.7, cumulative=False, density = True, label="FD")
    ax[1].hist(C_weight, bins = np.arange(0, 0.3, 0.002), alpha = 0.7, cumulative=False, density = True, label="Random C")
    ax[1].hist(BC_weight, bins = np.arange(0, 0.3, 0.002), alpha = 0.7, cumulative=False, density = True, label="Random BC")


    ax[0].legend()
    ax[1].legend()
    fig.savefig(f"log/{uuid_str}.png", bbox_inches='tight')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", type=str, help="uuid of the experiment")
    parser.add_argument("--p", type=float, default = 0.5)
    parser.add_argument("--times", type=int, default=2**12)
    parser.add_argument("-g", type = int)
    parser.add_argument("-d", type = int)
    parser.add_argument("--m1", type = int)
    parser.add_argument("--m2", type = int)
    parser.add_argument("--r2", type = int)
    parser.add_argument("--read", type=str, help="read from a file")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.uuid:
        H_s, R_s = load_data(args.uuid)
        FD = H_s.column_space().T
        C = R_s[:, args.g+args.d:]

        if args.read:
            with open(f"log/{args.uuid}_weight.json", "r") as f:
                FD_weight, C_weight = json.load(f)
        else:
            FD_weight = weight_distribution(FD, times = args.times, p = args.p)
            C_weight = weight_distribution(C, times = args.times, p = args.p)

            with open(f"log/{args.uuid}_weight.json", "w") as f:
                json.dump([FD_weight, C_weight], f)

        plt.hist(FD_weight, bins = np.arange(0.1, 0.3, 0.002), alpha = 0.7, cumulative=False, density = True, label="FD")
        plt.hist(C_weight, bins = np.arange(0.1, 0.3, 0.002), alpha = 0.7, cumulative=False, density = True, label="C")
        # plt.hist(R_s_weight, alpha = 0.7, cumulative=True, density = True, label="R_s")
        plt.legend()
        plt.savefig(f"log/{args.uuid}_p={args.p}.png")

    if args.compare:
        compare_various_codes()