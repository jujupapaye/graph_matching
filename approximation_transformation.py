"""

 Fonctions qui transforme une matrice p avec des coefficients entre 0 et 1
 en matrice de permutation de différentes manières

"""

import numpy as np
from munkres import *


def transformation_permutation(p):
    """
    Transforme une matrice avec des floats en une matrice de permutation
    en cherchant les max et en mettant à zero les lignes / colonnes
    :param p:
    :return:
    """
    n = p.shape[0]
    result = np.ones((n, 2), int)
    new = p.copy()
    for i in range(n):
        max = new.max()
        argmax = new.argmax()
        c = argmax % n
        l = int(argmax/n)
        new[l, c] = -1
        result[i][0] = l
        result[i][1] = c
        for j in range(n):
            if new[j, c] != -1:
                new[j, c] = 0
            if new[l, j] != -1:
                new[l, j] = 0
    return (-1) * new, result


def transformation_permutation_hungarian(p):
    """
    transformation d'une matrice p rempli de valeurs entre 0 et 1
    en une matrice de permutation grace à l'algorithme hongrois
    """
    m = Munkres()
    new = -1*p
    indexes = m.compute(new)
    res = np.zeros((p.shape[0], p.shape[0]))
    t = np.zeros((p.shape[0], 2))
    ind = 0
    for i, j in indexes:
        res[i, j] = 1
        t[ind, 0] = i
        t[ind, 1] = j
        ind += 1
    return res, t
