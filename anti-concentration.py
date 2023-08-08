import numpy as np
import galois
from time import time
from lib.construction import stabilizer_construction, qrc_construction
from lib.hypothesis import hypothesis_test, correlation_function
from lib.utils import get_H_s, rank
from lib.parallel import *
import json
import argparse, os

GF = galois.GF(2)

def moment(H):
    n = H.shape[1]
    d = GF.Random(n)
    # print(correlation_function_magnitude(H, d))
    H_d = get_H_s(H, d)
    g = rank(H_d.T @ H_d)

    return 2**(-g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--n-proc", type=int, default=-1, help="number of processes to run in parallel. Default is -1, meaning using all available CPUs.")
    parser.add_argument("--type", choices=["stab", "qrc"], default="stab")

    args = parser.parse_args()
    print(args)

    n_proc = os.cpu_count() if args.n_proc == -1 else args.n_proc
    global_settings(processes = n_proc) # set the number of processes to run in parallel

    g = 2
    q = 79
    
    tick = time()
    res = {}
    for n in range(100, 150, 2):
        res[n] = []
        for _ in range(10):
            m = n + 50
            if args.type == "stab":
                H, s = stabilizer_construction(n, m, g)
            elif args.type == "qrc":
                H, s = qrc_construction(n, m, q)
            print(H.shape)
            moment_list = tpmap(Task(moment), [H] * 10**4)
            
            res[n].append(np.mean(moment_list))

    with open(f"./data/moment_{args.type}.json", "w") as f:
        json.dump(res, f)

    print(f"Time elapsed: {(time() - tick)/3600:.2f} h")


    
    

