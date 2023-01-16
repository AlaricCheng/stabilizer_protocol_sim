import argparse
import galois, lib
from lib.utils import bias
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
from lib.parallel import *
from time import time


GF = galois.GF(2)

def merge_dict(dicts: list[dict]) -> dict:
    '''
    Merge a list of dictionaries. `dicts` must have the same keys.
    Example:
        >>> merge_dict([{'10': 100, '20': 250}, {'10': 39, '20': 97}])
        {10: [100, 39], 20: [250, 97]}
    '''
    merged_dict = {}
    for key in dicts[0]:
        merged_dict[int(key)] = [ele[key] for ele in dicts] # merge the values of the same key

    return merged_dict


def attack_general(n, thres = 15, nullity_range = (6, 10), g_thres = None) -> dict:
    '''
    General linearity attack.
    Args:
        n (int): number of columns before adding column redundancy
        thres (int): threshold for the number of checking secrets in the log base 2. Eg, if `thres = 15`, then the maximum number of checking secrets is `2**15 = 32768`.
        nullity_range (tuple): the minimum and maximum value of n-m/2. Default is (6, 10).
        g_thres (int): the threshold for rank of the Gram matrix. Default is None (which means 5)
    Return:
        data (dict): a dictionary with (n-m/2) as the key and a tuple of the number of checking secrets and the verification result as the value. Example: {5: (128, True), 6: (128, False), 7: (128, False), 8: (128, False), 9: (128, False)}
    '''
    data = {}
    for exp_nullity in range(nullity_range[0], nullity_range[1]):
        H, s = lib.generate_stab_instance(n, 2, exp_nullity = exp_nullity)
        LA = lib.LinearityAttack(H)
        X, c = LA.classical_sampling(5000, budget=2**thres, g_thres=g_thres, require_count=True)
        data[exp_nullity] = (c, lib.hypothesis_test(s, X, bias(LA.get_Gs_rank(s))))

    return data


def collect_data(n):
    '''
    Collect data for the figure.
    '''
    def helper(n):
        '''
        Helper function for parallel computing.
        '''
        data = []
        for i in range(3):
            data.append(attack_general(n, thres = args.thres + i, nullity_range = args.nullity_range, g_thres = args.g_thres))

        with open("tmp.log", "w") as f:
            json.dump(merge_dict(data), f)

        return merge_dict(data)

    data = tpmap(Task(helper), [n] * args.rep, desc = f"n = {n}")

    with open(f"./data/linearity_attack_{n}.json", "w") as f:
        json.dump(data, f)


def get_succ_prob(data: list) -> 'np.ndarray':
    '''
    Parse the data to get success probabilities. 
    Args:
        data (list): Each element of data is a list consisting of tuples of the form `(count, success)`.
    Example:
        >>> data = [[(32, 0), (64, 1)], [(32, 0), (64, 1)], [(32, 1), (64, 0)]]
        >>> get_succ_prob(data)
        [[32, 0.333], [64, 0.667]]
    '''
    data = np.array(data)
    res = []
    m = len(data[0])
    for i in range(m):
        tmp = np.array([ele[i] for ele in data])
        res.append(np.mean(tmp, axis = 0))

    return np.array(res)


def draw_fig(n):
    '''
    Draw the figure.
    '''
    with open(f"./data/linearity_attack_{n}.json", "r") as f:
        data = json.load(f)
        data = merge_dict(data)
        succ_prob = {k: get_succ_prob(v) for k, v in data.items()}
        # succ_prob is of the form {rd_col: np.ndarray}, where each row of the np.ndarray is of the form [count, success probability]
    print(succ_prob)
    fig, ax = plt.subplots(figsize = (6, 4))

    for i in range(3):
        tmp = {}
        for k, v in succ_prob.items():
            tweak = 0.000
            tmp[k] = v[(v[:, 0] == 2**(args.thres + i))][0, 1] + tweak
        x_ticks = np.fromiter(tmp.keys(), dtype = float) + 0.04*(i-1)
        ax.bar(x_ticks, tmp.values(), label = f"budget = {2**(args.thres+i)}", bottom = -tweak, width = 0.04)
    ax.set_xlabel("Max. of $n - \\frac{m}{2}$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Fraction of hacked instances")
    ax.set_ylim(-0.1, 1)
    ax.legend()

    fig.savefig(f"./fig/linearity_attack_{n}.{args.format}", bbox_inches = "tight")


def test(n):
    tick = time()
    H, s = lib.generate_stab_instance(n, 2, exp_nullity=7, verbose=args.verbose)
    print("Generation time:", time() - tick)
    if args.verbose:
        print("True secret:", s)
    
    tick = time()
    LA = lib.LinearityAttack(H)
    X, c = LA.classical_sampling(5000, budget=2**args.thres, g_thres=args.g_thres, require_count=True, independent_candidate=True, verbose=args.verbose)
    print((c, lib.hypothesis_test(s, X, bias(LA.get_Gs_rank(s)))))
    print("Attack time:", time() - tick)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--n-proc", type=int, default=3, help="number of processes to run in parallel. Default is 3.")
    parser.add_argument("--data", action = "store_true", help = "collect data (default: False)")
    parser.add_argument("--fig", action = "store_true", help = "generate figures (default: False)")

    parser.add_argument('n', type=int, help='number of columns before adding column redundancy')
    parser.add_argument('--thres', type=int, default=15, help='threshold for the number of checking secrets in the log base 2. Eg, if `thres = 15`, then the maximum number of checking secrets is `2**15 = 32768`.')
    parser.add_argument("--rep", type = int, default=100, help = "Number of repetitions for attacking each instance. Default is 100.")
    parser.add_argument("--nullity-range", type = int, nargs=2, default=[6, 10], help = "the minimum and maximum value of n-m/2. Default is (6, 10).")
    parser.add_argument("--g-thres", type = int, default = None, help = "the threshold for rank of the Gram matrix. Default is None (which means 5)")

    parser.add_argument("--test", action = "store_true", help = "test the correctness of the code (default: False)")
    parser.add_argument("-v", "--verbose", action = "store_true", help = "verbose mode (default: False)")

    parser.add_argument("--format", type = str, default = "svg", help = "format of the figure. Default is svg.")

    args = parser.parse_args()

    global_settings(processes = args.n_proc) # set the number of processes to run in parallel

    if args.data:
        collect_data(args.n)

    if args.fig:
        draw_fig(args.n)

    if args.test:
        for _ in range(args.rep):
            test(args.n)
    