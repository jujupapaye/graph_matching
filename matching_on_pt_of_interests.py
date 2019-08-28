import convex_multi_hsic as multihsic
import networkx as nx
import show_results_on_sphere as sh
import util
import numpy as np
import approximation_transformation as transfo


if __name__ == '__main__':
    K_list, graph_list = load_data.load_graph_and_kernels(5)

    for i in range(len(K_list)):
        K_list[i] = util.centered_matrix(K_list[i])
        K_list[i] = util.normalized_matrix(K_list[i])

    interet = 1   # choix du point d'interet où on regarde les pits

    if interet == 1:
        pt_interest = [-43, 6, 89]
        rayon_limite = 80
    else:
        pt_interest = [-48, 8, -87]
        rayon_limite = 60

    list_pits_to_remove = list()

    for g in range(len(graph_list)):
        for p in range(graph_list[g].number_of_nodes()):
            xp = graph_list[g].nodes[p]['coord'][0]
            yp = graph_list[g].nodes[p]['coord'][1]
            zp = graph_list[g].nodes[p]['coord'][2]
            new = list()
            list_pits_to_remove.append(new)
            if util.dist_on_sphere(100, pt_interest[0], pt_interest[1], pt_interest[2], xp, yp, zp) > rayon_limite:
                list_pits_to_remove[g].append(p)

    for k in range(len(K_list)):
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=0)
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=1)
        for node in list_pits_to_remove[k]:
            graph_list[k].remove_node(node)

    new_K_list = list()
    new_graph_list = list()

    for i in range(len(K_list)):
        if K_list[i].shape[0] == 11:
            new_K_list.append(K_list[i])
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
            new_graph_list.append(graph_list[i])

    nb_patients = len(new_K_list)
    nb_pits = new_K_list[0].shape[0]

    perms = np.zeros((nb_patients, nb_pits, nb_pits))
    perms[0] = np.eye(nb_pits)

    for p in range(1, len(perms)):
        # perms[p] = util.init_eig(new_K_list[0], new_K_list[p], nb_pits)
        perms[p] = util.init_random(nb_pits)   # initialisation des permutations

    # parameters of the gradient descent
    c, mu, mu_min, it, nb_tour = 1, 1e-43, 1e-43, 1500, 3  # params pour 25 graphes de taille 11
    # c, mu, mu_min, it, nb_tour = 1, 1e-10, 1e-30, 500, 3

    init = perms.copy()
    perms_opt = multihsic.estimate_perms(new_K_list, perms, c, mu, mu_min, it, nb_tour)

    t = np.zeros((nb_patients, nb_pits, 2))
    res = np.zeros((nb_patients, nb_pits, nb_pits))

    for i in range(1, nb_patients):
        res[i], t[i] = transfo.transformation_permutation_hungarian(perms_opt[i])  # transformation du résultat en matrices de permutations
    sh.show_sphere(t, new_graph_list)

