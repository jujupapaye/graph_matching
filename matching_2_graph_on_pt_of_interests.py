import convex_simple_hsic as hsic
import networkx as nx
import show_results_on_sphere as sh
import util
import numpy as np
import approximation_transformation as transfo
import branch_and_bound as branch
import load_graph_and_kernel as load_data
import time


def calcul_fobj(K0, K1, p):
    return np.linalg.norm(K0 @ p.T - p.T @ K1.T) ** 2, p.copy()


if __name__ == '__main__':
    K_list, graph_list = load_data.load_graph_and_kernels(5)

    for i in range(len(K_list)):
        K_list[i] = util.centered_matrix(K_list[i])
        K_list[i] = util.normalized_matrix(K_list[i])

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

    s0 = 30   # choix des graphes à comparer (entre 0 et 133)
    s1 = 2

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
    mu_min = 1e-8
    it = 1000
    c = 1
    nb_test = 300

    print("Paramètre mu/mu_min/it/c/nb_test:", mu, mu_min, it, c, nb_test)

    init = util.init_eig(K0, K1, nb_pits)
    res = hsic.estimate_perm(K0, K1, init.copy(), c, mu, mu_min, it)  # méthode minimisation convex
    t = transfo.transformation_permutation_hungarian(res)
    sorted_indices = t[0].argmax(axis=1)  # on récupère les indices où il y a un 1 pour toutes les lignes
    min_obj, p_min = calcul_fobj(K0, K1, t[0])
    print("MIN", min_obj)

    for i in range(nb_test):   # on fait plusieurs fois
        init = util.init_random(nb_pits)
        res = hsic.estimate_perm(K0, K1, init.copy(), c, mu, mu_min, it)  # méthode minimisation convex
        t = transfo.transformation_permutation_hungarian(res)
        sorted_indices = t[0].argmax(axis=1)  # on récupère les indices où il y a un 1
        perm = t[0].copy()
        obj = calcul_fobj(K0, K1, perm)[0]
        if obj < min_obj:
            min_obj = obj
            p_min = perm.copy()   # on choisit la permutation où est la fonction est au minimum
            print("min", min_obj)

    match = p_min.argmax(axis=1)
    sh.show_sphere_for_2(match, g0, g1)  # visualisation des résultats sur sphere

    init2 = util.init_eig(K0, K1, nb_pits)
    debut = time.clock()
    constraint = branch.branch_and_bound(K0, K1, init2, c=1, mu=1, mu_min=1e-7, it=300)  # méthode du branch and bound
    fin = time.clock()
    print("time:", fin-debut, " sec")

    match = np.zeros(len(constraint[0]))
    for i, j in constraint[0]:
        match[i] = j

    p = np.zeros((nb_pits, nb_pits))
    for i in range(match.shape[0]):
        p[i, int(match[i])] = 1

    obj = calcul_fobj(K0, K1, p)

    sh.show_sphere_for_2(match, g0, g1)

    # match = np.array([7,3,9,8,2,4,1,6,5,0])   # b and b for K[0] et K[2] de taille 10 sur point -43, 6, 89 rayon 80
    # c=1, mu=1, mu_min=1e-7, it=500   obj = 0.78864

    # match = np.array([2., 4., 7., 1., 9., 5., 0., 8., 6., 3.]) for K[2] et K[3] de taille 10 sur point -43, 6, 89 rayon 80
    # c=1, mu=1, mu_min=1e-7, it=300   obj =0.5576386506612289
    # match = np.array([3, 0, 2, 1, 4, 5, 6, 7, 9, 8]) -> meilleur match a la main pourtant obj = 1.34
    # match =np.array([7., 0., 2., 3., 6., 5., 1., 8., 9., 4.]) -> b&b avec noyau 3 obj =0.9484707644667182
    # match = np.array([6., 0., 3., 2., 9., 5., 8., 1., 7., 4.])-> b&b avec noyau 1  obj=0.494461710728216
    # match = np.array([6., 0., 3., 2., 9., 5., 8., 1., 7., 4.]) np.noyau 0 obj=0.5251326515258705