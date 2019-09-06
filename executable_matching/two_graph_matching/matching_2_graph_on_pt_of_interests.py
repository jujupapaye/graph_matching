"""
Test matching de 2 graphes sur des points d'interets
"""

from hsic import convex_simple as convex_simple_hsic, branch_and_bound as branch
import networkx as nx
from tools import util, approximation_transformation as transfo, metric, load_graph_and_kernel as load_data, \
    show_results as sh
import numpy as np
import time

if __name__ == '__main__':
    noyau = 5  # à changer selon le noyau qu'on veut
    noyaux = ["structure + coordonnées + profondeur", "coordonnées + profondeur ", "structure + profondeur",
              "structure + coordonnées", "stucture", "coordonnées", "profondeur"]
    K_list, graph_list = load_data.load_graph_and_kernels(noyau)  # coordinate kernel

    for i in range(len(K_list)):
        K_list[i] = util.normalized_matrix(K_list[i])
        K_list[i] = util.centered_matrix(K_list[i])

    interet = 1  # choix du point d'interet où on regarde les pits

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
                list_pits_to_remove[g].append(p)   # liste des pits (noeuds) à enlever dans le graphe car ils ne sont pas dans notre zone d'interet

    # on garde que les graphes et les noyaux associés dans le rayon de la zone d'interet
    for k in range(len(K_list)):
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=0)
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=1)
        for node in list_pits_to_remove[k]:
            graph_list[k].remove_node(node)

    new_graph_list = graph_list.copy()

    for i in range(len(graph_list)):
        new_graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])  # renommage des noeuds

    s0 = 20  # choix des graphes à comparer (entre 0 et 133)
    s1 = 33

    k0 = K_list[s0]
    k1 = K_list[s1]

    nb_pits = max(k0.shape[0], k1.shape[1])

    # préparation des données pour que les matrices ait la même taille
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

    g0 = new_graph_list[s0]
    g1 = new_graph_list[s1]

    nb_pits = K0.shape[0]

    # parameters for gradient descent
    mu = 1
    mu_min = 1e-5
    it = 1000
    c = 1
    nb_test = 1
    print("Comparaison des graphes", s0, "et", s1)
    print("Noyau :", noyaux[noyau])
    print("Convex Kernelized Sorting éxécuté", nb_test, "fois")
    print("Paramètre mu/mu_min/it/c:", mu, mu_min, it, c)

    init = util.init_eig(K0, K1, nb_pits)
    res = convex_simple_hsic.estimate_perm(K0, K1, init.copy(), c, mu, mu_min, it)  # méthode minimisation convex
    t = transfo.transformation_permutation_hungarian(res)
    sorted_indices = t[0].argmax(axis=1)  # on récupère les indices où il y a un 1 pour toutes les lignes
    min_obj, p_min = convex_simple_hsic.calcul_fobj(K0, K1, t[0])

    for i in range(nb_test):   # on fait plusieurs fois
        init = util.init_random(nb_pits)
        res = convex_simple_hsic.estimate_perm(K0, K1, init.copy(), c, mu, mu_min, it)  # méthode minimisation convex
        t = transfo.transformation_permutation_hungarian(res)
        sorted_indices = t[0].argmax(axis=1)  # on récupère les indices où il y a un 1
        perm = t[0].copy()
        obj = convex_simple_hsic.calcul_fobj(K0, K1, perm)[0]
        if obj < min_obj:
            min_obj = obj
            p_min = perm.copy()   # on choisit la permutation où est la fonction est au minimum

    match = p_min.argmax(axis=1)
    print("Fonction objectif:", min_obj)
    print("Moyenne des distances géodésique:", metric.metric_geodesic_for_2(match, g0, g1))
    sh.show_sphere_for_2(match, g0, g1)  # visualisation des résultats sur sphere
    sh.show_graph_for_2(match, g0, g1)

    print("Branch and bound en cours, cela peut durer un moment...")
    init2 = util.init_eig(K0, K1, nb_pits)
    debut = time.clock()
    constraint = branch.branch_and_bound(K0, K1, init2, c=1, mu=1, mu_min=1e-7, it=300)  # méthode du branch and bound
    fin = time.clock()
    print("Temps d'éxécution:", fin-debut, " sec")

    match = np.zeros(len(constraint[0]))
    for i, j in constraint[0]:
        match[i] = j

    p = np.zeros((nb_pits, nb_pits))
    for i in range(match.shape[0]):
        p[i, int(match[i])] = 1

    obj = convex_simple_hsic.calcul_fobj(K0, K1, p)
    print("Fonction objectif : ", obj[0])
    print("Moyenne des distances géosédiques", metric.metric_geodesic_for_2(match, g0, g1))
    sh.show_sphere_for_2(match, g0, g1)
    sh.show_graph_for_2(match, g0, g1)

