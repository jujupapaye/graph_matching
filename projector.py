##################

# Les projecteurs et leurs fonctions utiles

##################


import numpy as np


# projecteur qui met toutes les valeurs négatives d'une matrice à 0
def neg_projector(beta, mu):
    res = np.maximum(beta, np.zeros(beta.shape[0]))
    return res


# F est un filtre des valeurs fixé , renvoie une matrice "filtré" non négative
def create_projector_positivite_fixation(F, M):
    def projector_approx_matrix(beta, mu):
        beta = neg_projector(beta, mu)
        for i in range(beta.shape[0]):
            for j in range(beta.shape[1]):
                if F[i, j] == 0:  # si la valeur est fixé
                    beta[i, j] = M[i, j]
        return beta
    return projector_approx_matrix


# c les contraintes et n taille de la permutation souhaitée,
# renvoie la permutation correspondant aux contraintes c (avec des 0 partout où il n'y a pas de contraintes) et des 1 où il ya une une contrainte
def perm_with_constraint(c, n):
    perm = np.zeros((n, n))
    for (l, c) in c:
        perm[l, c] = 1
    return perm


# création d'un filtre selon les contraintes c, n étant la taille de la permutation voulu
def creation_filtre(c,n):
    perm = np.ones((n,n))   # initialisation en une matrice de 1
    for (l, c) in c:    # pour tout les conraintes (endroit où il y a un 1)
        perm[l, :] = 0   # on met des zeros sur les lignes et les colonnes
        perm[:, c] = 0
    return perm


# projecteur pour le branch and bound (negativité + valeurs fixés)
def create_projector_neg_and_fixed(constraints):
    def projector_neg_and_fixed(beta, mu):
        beta = neg_projector(beta, mu)
        perm = perm_with_constraint(constraints, beta.shape[0])
        filtre = creation_filtre(constraints, beta.shape[0])
        res = perm + beta*filtre
        return res
    return projector_neg_and_fixed
