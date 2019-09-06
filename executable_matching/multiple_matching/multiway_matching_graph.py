"""
Tests du matching de plusieurs graphes de pits
"""

from hsic import convex_multi as multihsic
from tools import util, approximation_transformation as transfo, load_graph_and_kernel as load_data, \
    show_results as sh
import numpy as np

if __name__ == '__main__':
    K_list, graph_list = load_data.load_graph_and_kernels(5)

    for i in range(len(K_list)):
        K_list[i] = util.centered_matrix(K_list[i])
        K_list[i] = util.normalized_matrix(K_list[i])

    new_K_list = []
    new_graph_list = []

    nb_pits = 86
    nb_graph = 10
    nb = 0

    for i in range(len(K_list)):
        if K_list[i].shape[0] == nb_pits and nb < nb_graph:
            new_graph_list.append(graph_list[i])
            new_K_list.append(K_list[i])
            nb += 1

    perms = np.zeros((nb_graph, nb_pits, nb_pits))
    perms[0] = np.eye(nb_pits)

    for p in range(1, len(perms)):
        # perms[p] = util.init_eig(K_list[0], K_list[p], nb_pits)
        perms[p] = util.init_random(nb_pits)   # initialisation des permutations

    # parameters of the gradient descent
    # c, mu, mu_min, it, nb_tour = 1, 1e-18, 1e-18, 500, 3    # params pour 3 graphes
    # c, mu, mu_min, it, nb_tour = 1, 1e-20*2.5, 2.5*1e-20, 500, 3   # params pour 5 graphes
    c, mu, mu_min, it, nb_tour = 1, 1e-28*1.5, 1e-28*1.5, 500, 3  # params pour 10 graphes

    init = perms.copy()
    perms_opt = multihsic.estimate_perms(new_K_list, perms, c, mu, mu_min, it, nb_tour)

    t = np.zeros((nb_graph, nb_pits, 2))
    res = np.zeros((nb_graph, nb_pits, nb_pits))

    for i in range(1, nb_graph):
        res[i], t[i] = transfo.transformation_permutation(perms_opt[i])
    sh.show_sphere(t, new_graph_list)
