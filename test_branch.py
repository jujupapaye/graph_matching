"""
Test matching pour 2 graphes avec branch and bound
"""
import load_graph_and_kernel as load_graph
import branch_and_bound as branch
import util
import show_results_on_sphere as sh
import numpy as np


def calcul_fobj(K0, K1, p):
    return np.linalg.norm(K0 @ p.T - p.T @ K1.T) ** 2, p.copy()


if __name__ == '__main__':
    K_list, graph_list = load_graph.load_graph_and_kernels(5) # coordonate kernel

    s0 = 45  # 17
    s1 = 12  # 28

    g0 = graph_list[s0]
    g1 = graph_list[s1]

    k0 = K_list[s0]
    k1 = K_list[s1]
    k0 = util.normalized_matrix(k0)  # normalisation des données
    k1 = util.normalized_matrix(k1)
    nb_pits = max(k0.shape[0], k1.shape[0])


    # préparation des données pour que les matrices à comparer ait la même taille
    # ajout de "faux" pit qui ne ressemble qu'à lui même (que des 0 sauf un 1 sur sa colonne)
    K0 = np.eye(nb_pits)
    K1 = np.eye(nb_pits)

    if k0.shape[0] == nb_pits:
        K0 = k0
    else:
        for i in range(k0.shape[0]):
            for j in range(k0.shape[1]):
                K0[i, j] = k0[i, j]

    if k1.shape[0] == nb_pits:
        K1 = k1
    else:
        for i in range(k1.shape[0]):
            for j in range(k1.shape[1]):
                K1[i, j] = k1[i, j]

    # parameters
    mu = 1
    mu_min = 1e-8
    it = 500
    c = 1

    print("Paramètre mu/mu_min/it/c/nb_test with branch and bound:", mu, mu_min, it, c)

    init = util.init_eig(K0, K1, nb_pits)
    res = branch.branch_and_bound(K0, K1, init, c, mu, mu_min, it)

    sh.show_sphere_for_2(match, g0, g1)  # visualisation des resultats
