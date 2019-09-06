"""
Test matching pour 2 graphes
"""

import convex_simple_hsic as hsic
import util
import show_results_on_sphere as sh
import numpy as np
import approximation_transformation as transfo
import load_graph_and_kernel as load_data
import metric


if __name__ == '__main__':
    noyau = 5  # à changer selon le noyau qu'on veut
    noyaux = ["structure + coordonnées + profondeur", "coordonnées + profondeur ", "structure + profondeur",
              "structure + coordonnées", "stucture", "coordonnées", "profondeur"]
    K_list, graph_list = load_data.load_graph_and_kernels(noyau)  # chargement des graphes et des matrices

    # normalisation et centrage des matrices
    for i in range(len(K_list)):
        K_list[i] = util.normalized_matrix(K_list[i])
        K_list[i] = util.centered_matrix(K_list[i])

    # numéros des sujets à comparer (entre 0 et 133)
    s0 = 0
    s1 = 1

    k0 = K_list[s0]  # matrice de gram
    k1 = K_list[s1]
    g0 = graph_list[s0]
    g1 = graph_list[s1]
    nb_pits = max(k0.shape[0], k1.shape[0])

    # préparation des données pour que les matrices aient la même taille
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

    # parameters for gradient descent
    mu = 1
    mu_min = 1e-8
    it = 1500
    c = 1
    nb_test = 500

    print("Comparaison des graphes", s0, "et", s1)
    print("Noyau :", noyaux[noyau])
    print("Convex Kernelized Sorting éxécuté", nb_test, "fois")
    print("Paramètre mu / mu_min / it / c : ", mu, mu_min, it, c)

    init = util.init_eig(K0, K1, nb_pits)
    res = hsic.estimate_perm(K0, K1, init, c, mu, mu_min, it)  # méthode minimisation convex
    t = transfo.transformation_permutation_hungarian(res)
    sorted_indices = t[0].argmax(axis=1)  # on récupère les indices où il y a un 1 pour toutes les lignes
    min_obj, p_min = hsic.calcul_fobj(K0, K1, t[0])

    for i in range(nb_test):
        init = util.init_random(nb_pits)
        res = hsic.estimate_perm(K0, K1, init.copy(), c, mu, mu_min, it)  # méthode minimisation convex
        t = transfo.transformation_permutation_hungarian(res)
        sorted_indices = t[0].argmax(axis=1)  # on récupère les indices où il y a un 1
        perm = t[0].copy()
        obj = hsic.calcul_fobj(K0, K1, perm)[0]
        if obj < min_obj:
            min_obj = obj
            p_min = perm.copy()

    match = p_min.argmax(axis=1)
    print("Moyenne des distances géodésique:", metric.metric_geodesic_for_2(match, g0, g1))
    sh.show_sphere_for_2(match, g0, g1)
    sh.show_graph_for_2(match, g0, g1)

