import json
import numpy as np
import matplotlib.pyplot as plt

fontsize = 14

def proc_qrc_kernel(q):
    with open(f"./data/qrc_kernel_{q}.json", "r") as f:
        res = json.load(f)
    
    return np.mean(res), np.std(res)


def proc_stab_kernel(n, m, g):
    with open(f"./data/stab_kernel_{n}_{m}_{g}.json", "r") as f:
        res = json.load(f)
    
    return np.mean(res), np.std(res)


def proc_stab_scheme_kernel(g):
    with open(f"./data/stab_scheme_kernel_{g}.json", "r") as f:
        res_raw = json.load(f)
    
    res = {}
    for k, v in res_raw.items():
        res[k] = np.mean(v), np.std(v)

    return res


def get_succ_prob(path):
    with open(path, "r") as f:
        raw_data = json.load(f)
    
    succ_prob = {}
    for k, v in raw_data.items():
        succ_prob[k] = np.mean([ele[0] for ele in v])

    return succ_prob


def get_num_checks(path):
    with open(path, "r") as f:
        raw_data = json.load(f)
    
    num_checks = {}
    for k, v in raw_data.items():
        num_checks[k] = np.mean([ele[1] for ele in v]), np.std([ele[1] for ele in v])

    return num_checks


def qrc_kernel_fig():
    qrc_kernel = []
    q_list = [103, 127, 151, 167]
    n_list = [int((q+1)/2) + q for q in q_list]
    for q in q_list:
        qrc_kernel.append(proc_qrc_kernel(q))
    
    qrc_kernel = np.array(qrc_kernel)
    
    exp_dim = np.array([(q+1)/2 for q in q_list])
    fig, ax = plt.subplots(figsize = (6, 4))

    ax.errorbar(n_list, qrc_kernel[:, 0], yerr = qrc_kernel[:, 1], fmt = "o", capsize = 3, label = "kernel dim (QRC)", color = "tab:blue")
    ax.scatter(n_list, exp_dim, label = "Exp. lower bound", marker = "*", color = "tab:orange", s = 50)

    ax.set_xlabel("Number of qubits $n$", fontsize = fontsize)
    ax.set_ylabel("Dim. of $\\mathrm{ker}(\\mathbf{G}_\\mathbf{d})$", fontsize = fontsize)
    ax.set_xticks(n_list)
    ax.tick_params(axis='both', labelsize=fontsize)

    ax.legend(fontsize = fontsize)
    fig.savefig("./fig/qrc_kernel.svg", bbox_inches = "tight")


def stab_kernel_fig():
    q_list = [103, 127, 151, 167]
    n_list = [int((q+1)/2) + q for q in q_list]
    stab_kernel_1 = []
    stab_kernel_3 = []

    for n in n_list:
        stab_kernel_1.append(proc_stab_kernel(n, n+50, 1))
        stab_kernel_3.append(proc_stab_kernel(n, n+50, 3))

    stab_kernel_1 = np.array(stab_kernel_1)
    stab_kernel_3 = np.array(stab_kernel_3)
    exp_dim = np.array([n - (n+50)/2 for n in n_list])

    fig, ax = plt.subplots(figsize = (6, 4))

    ax.errorbar(n_list, stab_kernel_1[:, 0], yerr = stab_kernel_1[:, 1], fmt = "o", capsize = 3, label = "kernel dim (g = 1)", color = "tab:blue")
    ax.errorbar(n_list, stab_kernel_3[:, 0], yerr = stab_kernel_3[:, 1], fmt = "o", capsize = 3, label = "kernel dim (g = 3)", color = "tab:green")
    ax.scatter(n_list, exp_dim, label = "Exp. lower bound", marker = "*", color = "tab:orange", s = 50)

    ax.set_xlabel("Number of qubits $n$", fontsize = fontsize)
    ax.set_ylabel("Dim. of $\\mathrm{ker}(\\mathbf{G}_\\mathbf{d})$", fontsize = fontsize)
    ax.set_xticks(n_list)
    ax.tick_params(axis='both', labelsize=fontsize)

    ax.legend(fontsize = fontsize)
    fig.savefig("./fig/stab_kernel.svg", bbox_inches = "tight")


def fix_SB_fig():
    q_list = [103, 127, 151, 167]
    
    succ_prob_SB = {}
    for q in q_list:
        path = f"./data/fix_SB_{q}.json"
        succ_prob_SB[q] = get_succ_prob(path)

    fig, ax = plt.subplots(figsize = (6, 4))

    for q in q_list:
        n_values = list(map(int, succ_prob_SB[q].keys()))
        succ_prob = [v for v in succ_prob_SB[q].values()]
        ax.plot(n_values, succ_prob, label = f"q = {q}")
        ax.scatter(q+15, 0, marker = "*", s = 60)

    ax.set_xlabel("Number of qubits $n$", fontsize = fontsize)
    ax.set_ylabel("Success probability", fontsize = fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    
    ax.legend(fontsize = fontsize)
    
    fig.savefig("./fig/fix_SB.svg", bbox_inches = "tight")


def compared_with_qrc_fig():
    q_list = [103, 127]

    succ_prob_stab = {}
    for q in q_list:
        path = f"./data/compared_with_qrc_{q}.json"
        succ_prob_stab[q] = get_succ_prob(path)

    fig, ax = plt.subplots(figsize = (6, 4))

    for q in q_list:
        n_values = list(map(int, succ_prob_stab[q].keys()))
        succ_prob = [v for v in succ_prob_stab[q].values()]
        ax.plot(n_values, succ_prob, label = f"m = {2*q}")

    ax.set_xlabel("Number of qubits $n$", fontsize = fontsize)
    ax.set_ylabel("Success probability", fontsize = fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    ax.legend(fontsize = fontsize)

    fig.savefig("./fig/compared_with_qrc.svg", bbox_inches = "tight")


def stab_scheme_fig():
    g_list = [[1, 1], [3, 3], [5, 5]]

    succ_prob_stab = {}
    for idx, g in enumerate(g_list):
        path = f"./data/stab_scheme_{g[0]}_{g[1]}.json"
        succ_prob_stab[idx] = get_succ_prob(path)

    fig, ax = plt.subplots(figsize = (6, 4))

    for idx, g in enumerate(g_list):
        n_values = list(map(int, succ_prob_stab[idx].keys()))
        succ_prob = [v for v in succ_prob_stab[idx].values()]
        ax.plot(n_values, succ_prob, label = f"g = {g[0]}, threshold = {g[1]}")

    ax.set_xlabel("Number of qubits $n$", fontsize = fontsize)
    ax.set_ylabel("Success probability", fontsize = fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    ax.legend(fontsize = fontsize)

    fig.savefig("./fig/stab_scheme.svg", bbox_inches = "tight")


def stab_scheme_kernel_fig():
    g_list = [1, 3, 5]

    stab_kernel = {}
    for g in g_list:
        stab_kernel[g] = proc_stab_scheme_kernel(g)
    
    fig, ax = plt.subplots(figsize = (6, 4))
    n_values = list(map(int, stab_kernel[1].keys()))
    exp_dim = np.array([n - 100 for n in n_values])

    ax.scatter(n_values, exp_dim, label = "Exp. lower bound", marker = "*", color = "lightcoral", s = 50)

    for g in g_list:
        dim = np.array([v[0] for v in stab_kernel[g].values()])
        dim_std = np.array([v[1] for v in stab_kernel[g].values()])
        ax.plot(n_values, dim, label = f"g = {g}")
        
        ax.fill_between(n_values, dim - dim_std, dim + dim_std, alpha = 0.2)

    ax.set_xlabel("Number of qubits $n$", fontsize = fontsize)
    ax.set_ylabel("Dim. of $\\mathrm{ker}(\\mathbf{G}_\\mathbf{d})$", fontsize = fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    ax.legend(fontsize = fontsize - 1)

    fig.savefig("./fig/stab_scheme_kernel.svg", bbox_inches = "tight")


if __name__ == "__main__":
    # qrc_kernel_fig()
    # stab_kernel_fig()
    # fix_SB_fig()
    stab_scheme_fig()

    stab_scheme_kernel_fig()

    # compared_with_qrc_fig()

    
