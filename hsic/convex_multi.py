"""

Fonctions utiles pour la minimisation convexe du HSIC pour n ensembles
argmin -HSIC(K0,K1,...KN) + c *contrainte    où contrainte = sommes des lignes/collonnes à 1
fonctions de création des fonctions objectif, gradient
et estimation des matrices de permutations à partir des n noyaux à comparer

"""
from minimization import convexminimization2 as cvm
from tools import projector as proj
import numpy as np


def create_objective_function(K, pi, c, ind, res1, res2, res3):
    """
    création de la fonction objectif (hsic)

    :param K: list of gram matrix to compare
    :param pi: list of permutation (initialisation matrix)
    :param c: hyperparameter for the constraint
    :param ind: indice of the pi
    :return: valeur de la fonction objectif pour les paramètres entrés
    """
    assert len(K) == len(pi)

    def objective_function(beta):
        n = K[0].shape[0]
        UN = np.ones((n, 1))
        constraints = np.linalg.norm(beta @ UN - UN)**2 + np.linalg.norm(beta.T @ UN - UN)**2
        r1 = res1 * (beta @ K[ind] @ beta.T)
        r2 = res2 @ (UN.T @ beta @ K[ind] @ beta.T @ UN)
        r3 = res3 * (beta @ K[ind] @ beta.T @ UN)
        r1 = UN.T @ r1 @ UN
        r3 = UN.T @ r3
        result = -1*(r1 + r2 - 2*r3)
        result += c*0.5*constraints
        return result
    return objective_function


def create_gradient_function(ind, K, pi, c, tmp1, tmp2, tmp3):
    """
    Création de la fonction gradient pour le HSIC multiple par rapport à pi[ind]
    :param ind: indice de pi par rapport auquel on calcule le gradient
    :param K: list of gram matrix to compare
    :param pi: list of permutation (initialisation matrix)
    :param c: hyperparameter for the constraint
    :return: gradient(HSIC)
    """
    assert len(K) == len(pi)

    def gradient_function(beta):
        n = K[0].shape[0]
        UN = np.ones((n, 1))
        constraints = np.ones((n, n)) @ beta + beta @ np.ones((n, n)) - 2 * np.ones((n, n))
        r1 = 2 * tmp1 @ beta @ K[ind]
        r2 = 2 * tmp2 * (np.ones((n, n)) @ beta @ K[ind])
        Q = tmp3 @ UN.T
        r3 = (Q + Q.T) @ beta @ K[ind]
        result = -1*(r1 + r2 - 2*r3)
        result += c * constraints
        return result
    return gradient_function


def estimate_perms(K, pi, c, mu, mu_min, it, nbtour):
    """
    Estimation des permutations pi par descente du gradient pour
    la minimisation du HSIC

    :param K: list of gram matrix to compare
    :param pi: list of permutation (initialisation matrix)
    :param c: hyperparameter for the constraint
    :param mu: initial step for the gradient descent
    :param mu_min: minimum step for the gradient descent
    :param it: number of iterations
    :param nbtour: number of tours that we do
    :return: pi optimal
    """
    indice = 1
    n = K[0].shape[0]
    pi[0] = np.eye(n)  # la première permutation sera toujours l'identité (on compare un ensemble avec lui même)
    UN = np.ones((n, 1))
    for i in range(1, n*nbtour):
        if i % nbtour != 0:
            res1 = np.ones((n, n))
            res2 = np.ones((1, 1))
            res3 = np.ones((n, 1))
            tmp1 = np.ones((n, n))
            tmp2 = np.ones((1, 1))
            tmp3 = np.ones((n, 1))
            for j in range(len(K)):
                if j != indice:
                    res1 = res1 * (pi[j] @ K[j] @ pi[j].T)
                    res2 = res2 @ (UN.T @ pi[j] @ K[j] @ pi[j].T @ UN)
                    res3 = res3 * (pi[j] @ K[j] @ pi[j].T @ UN)
                    tmp1 = tmp1 * (pi[j] @ K[j] @ pi[j].T)
                    tmp2 = tmp2 * (UN.T @ pi[j] @ K[j] @ pi[j].T @ UN)
                    tmp3 = tmp3 * (pi[j] @ K[j] @ pi[j].T @ UN)
            objective = create_objective_function(K, pi, c, indice, res1, res2, res3)
            gradient = create_gradient_function(indice, K, pi, c, tmp1, tmp2, tmp3)
            pi[indice], mu_est = cvm.monotone_fista_support(objective, gradient, pi[indice], mu, mu_min, it, proj.neg_projector)
            indice = (indice+1) % len(K)
    return pi
