import convex_simple_hsic as hsic
import networkx as nx
import show_results_on_sphere as sh
import util
import numpy as np
import approximation_transformation as transfo
import load_graph_and_kernel as load_data


def calcul_fobj(K0, K1, p):
    return np.linalg.norm(K0 @ p.T - p.T @ K1.T) ** 2, p.copy()


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

    for g in range(len(graph_list)):  # pour chaque graphe
        for p in range(graph_list[g].number_of_nodes()):  # pour chaque noeud
            xp = graph_list[g].nodes[p]['coord'][0]
            yp = graph_list[g].nodes[p]['coord'][1]
            zp = graph_list[g].nodes[p]['coord'][2]
            new = list()
            list_pits_to_remove.append(new)
            if util.dist_on_sphere(100, pt_interest[0], pt_interest[1], pt_interest[2], xp, yp, zp) > rayon_limite:
                list_pits_to_remove[g].append(p)   # liste des noeuds qui ne sont pas dans la zone d'interet

    # suppression des noeuds dans matrices de gram et dans les graphes
    for k in range(len(K_list)):
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=0)
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=1)
        for node in list_pits_to_remove[k]:
            graph_list[k].remove_node(node)

    pits_max = K_list[0].shape[0]
    for k in range(len(K_list)):    # trouver le nombre de pits maximum
        if K_list[k].shape[0] > pits_max:
            pits_max = K_list[k].shape[0]

    nb_pits = pits_max
    nb_graphs = len(K_list)

    new_K_list = np.zeros((nb_graphs, nb_pits, nb_pits))
    new_graph_list = list()

    # ajout des pits fictif pour que toute les matrices de gram fassent la même taille pits_max
    for i in range(nb_graphs):
        new_K_list[i] = np.eye(nb_pits)
        for j in range(K_list[i].shape[0]):
            for k in range(K_list[i].shape[0]):
                new_K_list[i, j, k] = K_list[i][j, k]
        graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        new_graph_list.append(graph_list[i])

    perms = np.zeros((nb_graphs, nb_pits, nb_pits))
    perms[0] = np.eye(nb_pits)

    # parameters of the gradient descent
    c, mu, mu_min, it, nb_tests = 1, 1, 1e-8, 1500, 300  # params pour 25 graphes de taille 11
    # c, mu, mu_min, it, nb_tour = 1, 1e-10, 1e-30, 500, 3

    init = perms.copy()
    perms_opt = np.zeros((nb_graphs, nb_pits, nb_pits))

    for i in range(1, nb_graphs):
        print(i, "/", nb_graphs)
        init = util.init_eig(new_K_list[0], new_K_list[i], nb_pits)
        perm = hsic.estimate_perm(new_K_list[0], new_K_list[i], init, c, mu, mu_min, it)
        t = transfo.transformation_permutation_hungarian(perm)
        min, perms_opt[i] = calcul_fobj(new_K_list[0], new_K_list[i], t[0])
        for t in range(nb_tests):
            init = util.init_random(nb_pits)
            perm = hsic.estimate_perm(new_K_list[0], new_K_list[i], init, c, mu, mu_min, it)
            t = transfo.transformation_permutation_hungarian(perm)
            obj = calcul_fobj(new_K_list[0], new_K_list[i], t[0])[0]
            if obj < min:
                perms_opt[i] = t[0].copy()
                min = obj

    match = np.zeros((nb_graphs, nb_pits, 2))
    res = np.zeros((nb_graphs, nb_pits, nb_pits))

    for i in range(1, nb_graphs):
        res[i], match[i] = transfo.transformation_permutation_hungarian(perms_opt[i])  # transformation du résultat en matrices de permutations
    sh.show_sphere(match, new_graph_list)

