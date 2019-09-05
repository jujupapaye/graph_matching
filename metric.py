import util
import numpy as np


def metric_geodesic(match, graphs):
    """
    Calcule la moyenne des distances géodésiques entre chaque points matchés entre eux
    :param match: list des matching
    :param graphs: list de graphes matchés
    :return: d
    """

    # tri des résultats pour faciliter l'interpretation après
    nb_graphs = len(graphs)
    for p in range(nb_graphs):
        match[p] = match[p][np.argsort(match[p][:, 1])]

    d = 0
    nb_pair = 0
    for m in range(1, len(match)):
        for i in range(len(match[m])):
            p1 = i
            p2 = match[m, i, 0]
            if graphs[0].has_node(p1) and graphs[m].has_node(p2):
                nb_pair += 1
                xA, yA, zA = graphs[0].node[p1]['coord']
                xB, yB, zB = graphs[m].node[p2]['coord']
                d += util.dist_on_sphere(100, xA, yA, zA, xB, yB, zB)
    return d / nb_pair


def metric_geodesic_for_2(match, g0, g1):
    """
    Calcule la moyenne des distances géodésiques entre chaque points matchés entre eux pour la comparaison de 2 graphes
    (ce qui change avec metric_geodesic c'est le type de match)
    :param match: liste des matching  où g0[match[i]] correspond à g1[i]
    :param g0: graphe
    :param g1: graphe
    :return: d
    """

    d = 0
    nb_pair = 0
    for m in range(0, len(match)):
        p1 = m
        p2 = match[m]
        if g0.has_node(p2) and g1.has_node(p1):
            nb_pair += 1
            xA, yA, zA = g0.node[p2]['coord']
            xB, yB, zB = g1.node[p1]['coord']
            d += util.dist_on_sphere(100, xA, yA, zA, xB, yB, zB)
    return d / nb_pair
