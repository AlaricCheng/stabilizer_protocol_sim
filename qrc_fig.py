import argparse
import galois, lib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from lib.construction import q_helper
from lib.parallel import *


GF = galois.GF(2)

def get_qrc_size_list(q):
    q_list = np.array(q_helper(1000))
    q_list = q_list[(q_list >= q)][:5]
    return q_list


def attack_qrc_original(q, thres = 15, lim = 40):
    start_pt, end_pt = int((q+1)/2 - lim/2), int((q+1)/2 + lim/2)
    
    data = {}
    for rd_col in range(start_pt, end_pt+1, 2):
        # print(rd_col)
        H, s = lib.generate_QRC_instance(q, q, rd_col)
        X, c = lib.QRCAttack(H).classical_sampling_original(10*H.shape[0], budget=2**thres, require_count = True)
        data[rd_col] = (c, lib.hypothesis_test(s, X, 0.854))
    return data

def helper_original(q):
    return attack_qrc_original(q, thres = args.thres, lim = args.lim)

def dict_to_arr(tmp):
    res = [[key] + value for key, value in zip(tmp.keys(), tmp.values())]
    return np.array(res)

def merge_dict(dicts: list[dict]) -> list:
    '''
    dicts contains dictionaries with the same keys. Merge the values into a list.
    '''
    merged_dict = {}
    for key in dicts[0]:
        merged_dict[key] = [ele[key] for ele in dicts]

    count_dict = {}
    for key in merged_dict:
        count_dict[key] = [ele[0] for ele in merged_dict[key]]

    succ_dict = {}
    for key in merged_dict:
        succ_dict[key] = np.mean([ele[1] for ele in merged_dict[key]])
    
    return dict_to_arr(count_dict), succ_dict 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--n-proc", type = int, default = 3, help = "number of processors used")

    parser.add_argument("-q", nargs="*", type = int, default=[7], help = "the start point of size parameters for QRC")
    parser.add_argument("--thres", type = int, default = 15, help = "budget is 2^thres")
    parser.add_argument("--rep", type = int, default=20, help = "Number of repetitions for attacking each instance")
    parser.add_argument("--lim", type = int, default=40, help = "the range of column redundancy")
    
    args = parser.parse_args()
    
    global_settings(processes = args.n_proc)
    q_list = args.q

    fig, ax = plt.subplots(ncols=2, figsize = (8, 4), constrained_layout = True)
    for q in q_list:
        tmp = tpmap(Task(helper_original), [q]*args.rep, desc = f"q = {q}")
        counts, succ_prob = merge_dict(tmp)
        ax[0].errorbar(counts[:, 0], np.mean(counts[:, 1:], axis = 1), yerr=np.std(counts[:, 1:], axis = 1), fmt = "^--", label = f"q = {q}")
        ax[1].plot(succ_prob.keys(), succ_prob.values(), "^--", label = f"q = {q}")

    ax[0].set_xlabel("Degree of column redundancy")
    ax[0].set_ylabel("Number of checked solutions")

    ax[1].set_xlabel("Degree of column redundancy")
    ax[1].set_ylabel("Success probability")
    ax[1].legend()

    fig.savefig(f"./fig/original_attack_{q_list}.svg")