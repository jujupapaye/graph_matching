"""

 Les projecteurs et leurs fonctions utiles

"""


import numpy as np


def neg_projector(beta, mu):
    """
    Projecteur mettant les valeurs negative de beta à 0
    :return: beta avec valeur negative à 0
    """
    res = np.maximum(beta, np.zeros(beta.shape[0]))
    return res


def perm_with_constraint(constraint, n):
    """
    Retourne une permutation avec des constraints
    :param constraint: list de tuple (x,y) où (x,y) est une contrainte
    :param n: taille de la permutation souhaitée
    :return: permutation perm de taille n avec perm[x,y] = 1 pour toute les contraintes
    """
    perm = np.zeros((n, n))
    for (l, c) in constraint:
        perm[l, c] = 1
    return perm


def creation_filtre(constraint, n):
    """
    Création d'un filtre selon les contraintes c
    :param constraint: list de tuple (x,y) où (x,y) est une contrainte
    :param n: taille du filtre voulu
    :return: filtre avec des 0 sur les lignes et les colonnes des contraintes
    """
    perm = np.ones((n, n))   # initialisation en une matrice de 1
    for (l, c) in constraint:    # pour tout les contraintes (endroit où il y a un 1)
        perm[l, :] = 0   # on met des zeros sur les lignes et les colonnes
        perm[:, c] = 0
    return perm


def create_projector_neg_and_fixed(constraints):
    """
        Projecteur mettant les valeurs negatives de beta à 0 avec un filtre (pour branch and bound)
        :param constraints: list de tuple (x,y) où (x,y) est une contrainte
        :return: beta avec ses valeurs fixé + projecteur negatif
    """
    def projector_neg_and_fixed(beta, mu):
        beta = neg_projector(beta, mu)
        perm = perm_with_constraint(constraints, beta.shape[0])
        filtre = creation_filtre(constraints, beta.shape[0])
        res = perm + beta*filtre
        return res
    return projector_neg_and_fixed
