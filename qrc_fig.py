import argparse
from typing import Tuple
import galois, lib
import numpy as np
import matplotlib.pyplot as plt
import json
from lib.parallel import *


GF = galois.GF(2)

def dict_to_arr(data: dict) -> 'np.ndarray':
    '''
    Convert a dictionary to a numpy array.
    '''
    res = [[key] + value for key, value in zip(data.keys(), data.values())]

    return np.array(res)


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


def parse_dicts(dicts: list[dict]) -> Tuple:
    '''
    Parse a list of dictionaries and return a tuple.
    Each dictionary in `dicts` should have the same keys, with the values of the form `(count, success)`. The keys are the degree of column redundancy.
    Example:
        >>> parse_dicts([{'10': (100, 0), '20': (250, 1)}, {'10': (39, 1), '20': (97, 1)}])
        >>> array([[10, 100, 39],[20, 250, 97]]), {10: 0.5, 20: 1.0}
    '''
    merged_dict = merge_dict(dicts)

    count_dict = {} # number of solutions checked
    for key in merged_dict:
        count_dict[key] = [ele[0] for ele in merged_dict[key]]

    succ_dict = {}
    for key in merged_dict:
        succ_dict[key] = np.mean([ele[1] for ele in merged_dict[key]])
    
    return dict_to_arr(count_dict), succ_dict 


def attack_qrc_original(q, thres = 15, lim = 60) -> dict:
    '''
    Generate data for the original qrc attack. 
    Args:
        q (int): the size parameter of the qrc
        thres (int): the threshold for the number of checking secrets in the log base 2. Eg, if `thres = 15`, then the maximum number of checking secrets is `2**15 = 32768`.
        lim (int): the maximum degree of column redundancy is `(q+1)/2 + lim`
    Return:
        data (dict): a dictionary with the degree of column redundancy as the key, and the value is a tuple of the form `(count, success)`, where `count` is the number of checking secrets, and `success` is a 0-1 value.
    '''
    start_pt, end_pt = int((q+1)/2), int((q+1)/2 + lim)
    
    data = {}
    for rd_col in range(start_pt, end_pt+1, 2):
        # print(rd_col)
        H, s = lib.generate_QRC_instance(q, q, rd_col)
        X, c = lib.QRCAttack(H).classical_sampling_original(5000, budget=2**thres, require_count = True)
        data[rd_col] = (c, lib.hypothesis_test(s, X, 0.854))

    return data


def helper_original(q) -> dict:
    '''helper function for attack_qrc_original'''
    return attack_qrc_original(q, thres = args.thres, lim = args.lim)


def collect_data_original(q_list):
    '''
    Dump the data for the original qrc attack to a json file.
    '''
    for q in q_list:
        data = tpmap(Task(helper_original), [q]*args.rep, desc = f"q = {q}")

        with open(f"./data/original_attack_{q}.json", "w") as f:
            json.dump(data, f)


def draw_fig_original(q_list):
    '''
    Draw figures for the original qrc attack.
    '''
    for q in q_list:
        fig, ax = plt.subplots(ncols=2, figsize = (8, 4), constrained_layout = True)

        with open(f"./data/original_attack_{q}.json", "r") as f:
            data = json.load(f)
            counts, succ_prob = parse_dicts(data)

        ax[0].errorbar(counts[:, 0], np.mean(counts[:, 1:], axis = 1), yerr=np.std(counts[:, 1:], axis = 1), fmt = "^--", color = "royalblue", label = f"q = {q}")
        ax[1].plot(succ_prob.keys(), succ_prob.values(), "^--", color = "royalblue", label = f"q = {q}")

        ax[0].set_xlabel("Degree of column redundancy")
        ax[0].set_ylabel("Number of checked solutions")

        ax[1].set_xlabel("Degree of column redundancy")
        ax[1].set_ylabel("Fraction of hacked instances")
        ax[1].legend()

        fig.savefig(f"./fig/original_attack_{q}.svg")


def attack_qrc_enhanced(q, thres = 15, lim = 60) -> dict:
    '''
    Attack QRC with enhanced classical sampling.
    Args:
        q (int): the size parameter of the qrc
        thres (int): the threshold for the number of checking secrets in the log base 2. Eg, if `thres = 15`, then the maximum number of checking secrets is `2**15 = 32768`.
        lim (int): the maximum degree of column redundancy is `(q+1)/2 + lim`
    '''
    start_pt, end_pt = int((q+1)/2), int((q+1)/2 + lim)

    data = {}
    for rd_col in range(start_pt, end_pt+1):
        # print(rd_col)
        H, s = lib.generate_QRC_instance(q, q, rd_col)
        X, c = lib.QRCAttack(H).classical_sampling_enhanced(5000, budget=2**thres, require_count=True)
        data[rd_col] = (c, lib.hypothesis_test(s, X, 0.854))

    return data


def helper_enhanced(q):
    '''
    Helper function for attack_qrc_enhanced
    Example:
        >>> helper_enhanced(31) # thres = 5, lim = 4
        {
            16: [(32, False), (64, True), (128, False), (256, True), (512, True)],
            18: [(32, False), (64, False), (128, True), (256, False), (512, True)], 
            20: [(32, False), (64, True), (128, False), (256, True), (512, False)]
        }
    '''
    data = []
    for i in range(3):
        data.append(attack_qrc_enhanced(q, thres = args.thres+i, lim = args.lim))

    with open("tmp.log", "w") as f:
        json.dump(merge_dict(data), f)
    
    return merge_dict(data)


def collect_data_enhanced(q_list):
    '''
    Dump the data for the enhanced qrc attack to a json file.
    '''
    for q in q_list:
        data = tpmap(Task(helper_enhanced), [q]*args.rep, desc = f"q = {q}")

        with open(f"./data/enhanced_attack_{q}.json", "w") as f:
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


def draw_fig_enhanced(q_list):
    '''
    Draw figures for the enhanced qrc attack.
    '''
    for q in q_list:
        fig, ax = plt.subplots(figsize = (6, 4))

        with open(f"./data/enhanced_attack_{q}.json", "r") as f:
            data = json.load(f)
            data = merge_dict(data)
            succ_prob = {k: get_succ_prob(v) for k, v in data.items()}
        # succ_prob is of the form {rd_col: np.ndarray}, where each row of the np.ndarray is of the form [count, success probability]

        for i in range(3):
            tmp = {}
            for k, v in succ_prob.items():
                tmp[k] = v[(v[:, 0] == 2**(args.thres+i))][0, 1] # get the success probability for the corresponding budget
            ax.plot(tmp.keys(), tmp.values(), "^--", label = f"thres = {2**(args.thres+i)}")
        ax.set_xlabel("Degree of column redundancy")
        ax.set_ylabel("Fraction of hacked instances")
        ax.legend()

        fig.savefig(f"./fig/enhanced_attack_{q}.svg")


def test(q):
    with open(f"./data/enhanced_attack_{q}.json", "r") as f:
        data = json.load(f)
        print(len(data))
        data = merge_dict(data)
        succ_prob = {k: get_succ_prob(v) for k, v in data.items()}
    print(succ_prob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type = str, help = "the type of attack ('original' or 'enhanced'")
    parser.add_argument("-p", "--n-proc", type = int, default = 3, help = "number of processors used")
    parser.add_argument("--data", action = "store_true", help = "collect data (default: False)")
    parser.add_argument("--fig", action = "store_true", help = "generate figures (default: False)")

    parser.add_argument("-q", nargs="*", type = int, default=[7], help = "the start point of size parameters for QRC")
    parser.add_argument("--thres", type = int, default = 15, help = "budget is 2^thres")
    parser.add_argument("--rep", type = int, default=30, help = "Number of repetitions for attacking each instance")
    parser.add_argument("--lim", type = int, default=60, help = "the range of column redundancy")

    parser.add_argument("--test", action = "store_true", help = "test the code (default: False)")
    
    args = parser.parse_args()
    
    global_settings(processes = args.n_proc) # set the number of processors used
    q_list = args.q

    if args.data:
        if args.type == "original":
            collect_data_original(q_list)
        elif args.type == "enhanced":
            collect_data_enhanced(q_list)

    if args.fig:
        if args.type == "original":
            draw_fig_original(q_list)
        elif args.type == "enhanced":
            draw_fig_enhanced(q_list)

    if args.test:
        test(151)


