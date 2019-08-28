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

    pt_interest1 = [-43, 6, 89]
    pt_interest2 = [-48, 8, -87]

    list_pits_to_remove = list()
    rayon_limite = 80

    for g in range(len(graph_list)):
        for p in range(graph_list[g].number_of_nodes()):
            xp = graph_list[g].nodes[p]['coord'][0]
            yp = graph_list[g].nodes[p]['coord'][1]
            zp = graph_list[g].nodes[p]['coord'][2]
            new = list()
            list_pits_to_remove.append(new)
            if util.dist_on_sphere(100, -43, 6, 89, xp, yp, zp) > rayon_limite:
            # if util.dist_on_sphere(100, -48, 8, -87, xp, yp, zp) > 60:
                list_pits_to_remove[g].append(p)   # liste des pits (noeuds) à enlever dans le graphe car ils ne sont pas dans notre zone d'interet

    for k in range(len(K_list)):
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=0)
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=1)
        for node in list_pits_to_remove[k]:
            graph_list[k].remove_node(node)

    new_graph_list = graph_list.copy()

    for i in range(len(graph_list)):
        new_graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])  # renommage des noeuds

    K0 = K_list[0]
    K1 = K_list[2]

    g0 = new_graph_list[0]
    g1 = new_graph_list[2]

    nb_pits = K0.shape[0]

    # parameters for gradient descent
    mu = 1
    mu_min = 1e-8
    it = 1000
    c = 1
    nb_test = 1

    print("Paramètre mu/mu_min/it/c/nb_test:", mu, mu_min, it, c, nb_test)

    init = util.init_eig(K0, K1, nb_pits)
    res = hsic.estimate_perm(K0, K1, init.copy(), c, mu, mu_min, it)  # méthode minimisation convex
    t = transfo.transformation_permutation_hungarian(res)
    sorted_indices = t[0].argmax(axis=1)  # on récupère les indices où il y a un 1 pour toutes les lignes
    min_obj, p_min = calcul_fobj(K0, K1, t[0])
    print("MIN", min_obj)

    for i in range(nb_test):
        init = util.init_random(nb_pits)
        res = hsic.estimate_perm(K0, K1, init.copy(), c, mu, mu_min, it)  # méthode minimisation convex
        t = transfo.transformation_permutation_hungarian(res)
        sorted_indices = t[0].argmax(axis=1)  # on récupère les indices où il y a un 1
        perm = t[0].copy()
        obj = calcul_fobj(K0, K1, perm)[0]
        if obj < min_obj:
            min_obj = obj
            p_min = perm.copy()
            print("min", min_obj)

    match = p_min.argmax(axis=1)
    sh.show_sphere_for_2(match, g0, g1)

    init2 = util.init_eig(K0, K1, nb_pits)
    debut = time.clock()
    constraint, l, u = branch.branch_and_bound(K0, K1, init2, c=1, mu=1, mu_min=1e-7, it=500)
    fin = time.clock()
    print("time:", fin-debut)


    match = np.array([7,3,9,8,2,4,1,6,5,0])   # b and b for K[0] et K[2] de taille 11 sur point -43, 6, 89 rayon 80
    # c=1, mu=1, mu_min=1e-7, it=500   obj = 0.78864