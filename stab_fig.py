import argparse
import galois, lib
from lib.utils import bias
import numpy as np
import matplotlib.pyplot as plt
import json
from lib.parallel import *


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


def attack_general(n, thres = 15, rd_range = (13, 26), g_thres = None) -> dict:
    '''
    General linearity attack.
    Args:
        n (int): number of columns before adding column redundancy
        thres (int): threshold for the number of checking secrets in the log base 2. Eg, if `thres = 15`, then the maximum number of checking secrets is `2**15 = 32768`.
        rd_range (tuple): the minimum and maximum value of n-m/2. Default is (13, 26).
        g_thres (int): the threshold for rank of the Gram matrix. Default is None (which means 5)
    Return:
        data (dict): a dictionary with (n-m/2) as the key and a tuple of the number of checking secrets and the verification result as the value. Example: {5: (128, True), 6: (128, False), 7: (128, False), 8: (128, False), 9: (128, False)}
    '''
    data = {}
    for rd_col_ext in range(rd_range[0], rd_range[1]):
        H, s = lib.generate_stab_instance(n, 3, rd_col_ext = rd_col_ext)
        LA = lib.LinearityAttack(H)
        X = LA.classical_sampling(5000, budget=2**thres, g_thres=g_thres)
        data[rd_col_ext] = (2**thres, lib.hypothesis_test(s, X, bias(LA.get_Gs_rank(s))))

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
            data.append(attack_general(n, thres = args.thres + i, rd_range = args.rd_range, g_thres = args.g_thres))

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
    
    fig, ax = plt.subplots(figsize = (6, 4))

    for i in range(3):
        tmp = {}
        for k, v in succ_prob.items():
            tmp[k] = v[(v[:, 0] == 2**(args.thres + i))][0, 1]
        ax.plot(tmp.keys(), tmp.values(), "^--", label = f"thres = {2**(args.thres+i)}")
    ax.set_xlabel("$n - \\frac{m}{2}$")
    ax.set_ylabel("Fraction of hacked instances")
    ax.legend()

    fig.savefig(f"./fig/linearity_attack_{n}.svg", bbox_inches = "tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--n-proc", type=int, default=3, help="number of processes to run in parallel. Default is 3.")
    parser.add_argument("--data", action = "store_true", help = "collect data (default: False)")
    parser.add_argument("--fig", action = "store_true", help = "generate figures (default: False)")

    parser.add_argument('n', type=int, help='number of columns before adding column redundancy')
    parser.add_argument('--thres', type=int, default=15, help='threshold for the number of checking secrets in the log base 2. Eg, if `thres = 15`, then the maximum number of checking secrets is `2**15 = 32768`.')
    parser.add_argument("--rep", type = int, default=100, help = "Number of repetitions for attacking each instance. Default is 100.")
    parser.add_argument("--rd-range", type = int, nargs=2, default=[13, 26], help = "the minimum and maximum value of n-m/2. Default is (13, 26).")
    parser.add_argument("--g-thres", type = int, default = None, help = "the threshold for rank of the Gram matrix. Default is None (which means 5)")

    args = parser.parse_args()

    global_settings(processes = args.n_proc) # set the number of processes to run in parallel

    if args.data:
        collect_data(args.n)

    if args.fig:
        draw_fig(args.n)
    