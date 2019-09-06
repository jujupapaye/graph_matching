"""
Test matching pour 2 graphes avec branch and bound
"""
from tools import util, metric, load_graph_and_kernel as load_graph, show_results_on_sphere as sh
import numpy as np
from hsic import convex_simple as convex_simple_hsic, branch_and_bound as branch

if __name__ == '__main__':
    noyau = 5  # à changer selon le noyau qu'on veut
    noyaux = ["structure + coordonnées + profondeur", "coordonnées + profondeur ", "structure + profondeur",
              "structure + coordonnées", "stucture", "coordonnées", "profondeur"]
    K_list, graph_list = load_graph.load_graph_and_kernels(noyau)

    # numéros des sujets à comparer (entre 0 et 133)
    s0 = 17
    s1 = 28

    g0 = graph_list[s0]
    g1 = graph_list[s1]

    k0 = K_list[s0]
    k1 = K_list[s1]
    k0 = util.normalized_matrix(k0)  # normalisation des données
    k1 = util.normalized_matrix(k1)
    k0 = util.centered_matrix(k0)
    k1 = util.centered_matrix(k1)
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
    mu_min = 1e-6
    it = 350
    c = 1

    print("Comparaison des graphes", s0, "et", s1)
    print("Noyau :", noyaux[noyau])
    print("Paramètre mu/mu_min/it/c with branch and bound:", mu, mu_min, it, c)

    init = util.init_eig(K0, K1, nb_pits)
    constraint = branch.branch_and_bound(K0, K1, init, c, mu, mu_min, it)

    # transformation du résultats pour la visualisation
    match = np.zeros(len(constraint[0]))
    for i, j in constraint[0]:
        match[i] = j

    p = np.zeros((nb_pits, nb_pits))
    for i in range(match.shape[0]):
        p[i, int(match[i])] = 1

    obj = convex_simple_hsic.calcul_fobj(K0, K1, p)
    print("Fonction objectif : ", obj[0])
    print("Moyenne des distances géosédiques", metric.metric_geodesic_for_2(match, g0, g1))
    sh.show_sphere_for_2(match, g0, g1)  # visualisation des resultats
