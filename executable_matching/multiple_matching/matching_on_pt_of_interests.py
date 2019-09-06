"""
Matching de plusieurs graphes sur centres d'interets
en comparant tous les graphes à un graphes modèle
"""

from hsic import convex_simple as convex_simple_hsic
import networkx as nx
from tools import util, approximation_transformation as transfo, metric, load_graph_and_kernel as load_data, \
    show_results as sh
import numpy as np

if __name__ == '__main__':
    noyau = 5   # à changer selon le noyau qu'on veut
    noyaux = ["structure + coordonnées + profondeur", "coordonnées + profondeur ", "structure + profondeur", "structure + coordonnées", "stucture", "coordonnées", "profondeur"]
    K_list, graph_list = load_data.load_graph_and_kernels(noyau)  # noyau coordonnées

    for i in range(len(K_list)):
        K_list[i] = util.normalized_matrix(K_list[i])
        K_list[i] = util.centered_matrix(K_list[i])

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
    indice_pit_max = 0
    for k in range(len(K_list)):    # trouver le nombre de pits maximum
        if K_list[k].shape[0] > pits_max:
            pits_max = K_list[k].shape[0]
            indice_pit_max = k

    nb_pits = pits_max
    nb_graphs = len(K_list)

    # on met en premier un graphe qui a le nombre de pit max pour le comparer avec tous les autres
    tmp = K_list[indice_pit_max].copy()
    K_list[indice_pit_max] = K_list[0].copy()
    K_list[0] = tmp
    graph_tmp = graph_list[indice_pit_max].copy()
    graph_list[indice_pit_max] = graph_list[0].copy()
    graph_list[0] = graph_tmp

    new_K_list = np.zeros((nb_graphs, nb_pits, nb_pits))
    new_graph_list = list()

    # ajout des pits fictifs pour que toutes les matrices de gram fassent la même taille (pits_max)
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
    c, mu, mu_min, it, nb_tests = 1, 1, 1e-5, 500, 300  # pt d'interet 2
    print("Comparaison de tous les graphes (134)")
    print("Noyau :", noyaux[noyau])
    print("Paramètres :c, mu, mu_min, it, nb_tests ",c, mu, mu_min, it, nb_tests)

    init = perms.copy()
    perms_opt = np.zeros((nb_graphs, nb_pits, nb_pits))

    for i in range(1, nb_graphs):
        print(i, "/", nb_graphs)
        init = util.init_eig(new_K_list[0], new_K_list[i], nb_pits)
        perm = convex_simple_hsic.estimate_perm(new_K_list[0], new_K_list[i], init, c, mu, mu_min, it)
        t = transfo.transformation_permutation_hungarian(perm)
        min, perms_opt[i] = convex_simple_hsic.calcul_fobj(new_K_list[0], new_K_list[i], t[0])
        for t in range(nb_tests):
            init = util.init_random(nb_pits)
            perm = convex_simple_hsic.estimate_perm(new_K_list[0], new_K_list[i], init, c, mu, mu_min, it)
            t = transfo.transformation_permutation_hungarian(perm)
            obj = convex_simple_hsic.calcul_fobj(new_K_list[0], new_K_list[i], t[0])[0]
            if obj < min:
                perms_opt[i] = t[0].copy()
                min = obj

    match = np.zeros((nb_graphs, nb_pits, 2))
    res = np.zeros((nb_graphs, nb_pits, nb_pits))

    for i in range(1, nb_graphs):
        res[i], match[i] = transfo.transformation_permutation_hungarian(perms_opt[i])  # transformation du résultat en matrices de permutations
    print("Geodesis metric : ", metric.metric_geodesic(match, new_graph_list))
    sh.show_sphere(match, new_graph_list)

