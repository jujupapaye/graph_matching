"""

 Les fonctions utiles pour la minimisation convexe du HSIC pour 2 ensembles
  argmin ||K @ X-X.T @ L.T||^2 + c *contrainte    où contrainte = sommes des lignes/colonnes à 1
 fonctions de création des fonctions objectif, gradient + estimation d'une permutation à partir de 2 matrices noyaux à comparer

"""

import numpy as np
import convexminimization2 as cvm
import projector as proj


def create_objective_function(K, L, pi, c):
    """

    :param K: gram matrix of the first set to compare
    :param L: gram matrix of the second set to compare
    :param pi: permutation to find
    :param c: hyperparameter
    :return:
    """
    def objective_function(beta):
        n = K.shape[0]
        UN = np.ones((n, 1))
        constraints = np.linalg.norm(beta @ UN - UN)**2 + np.linalg.norm(beta.T @ UN - UN)**2
        res = np.linalg.norm(K @ beta.T - beta.T @ L.T)**2 + c*constraints
        return res
    return objective_function


def create_gradient_function(K, L, pi, c):
    """

    :param K: gram matrix of the first set to compare
    :param L: gram matrix of the second set to compare
    :param pi: permutation to find
    :param c: hyperparameter
    :return:
    """
    def gradient_function(beta):
        n = K.shape[0]
        result = 2*(beta @ K.T @ K - L @ beta @ K) - 2*(L.T @ beta @ K.T - L.T @ L @ beta)
        constraints = 2 * (np.ones((n, n)) @ beta + beta @ np.ones((n, n)) - 2 * np.ones((n, n)))
        return result + c*constraints
    return gradient_function


# estime une permutation p pour comparer 2 objets (convex kernelized sorting)
# avec K et L matrices noyaux centrés des ensembles à comparer
# init la matrice d'initialisation de la descente du gradient, c l'hyperparamètre qui impose le poids des contraintes,
# it le nombre d'itérations à faire pour la descente
def estimate_perm(K, L, init, c, mu, mu_min, it):
    """

    :param K: gram matrix of the first set to compare
    :param L: gram matrix of the second set to compare
    :param init: initialisation matrix
    :param c: hyperparameter
    :param mu: initial step for the gradient descent
    :param mu_min: minimum step for the gradient descent
    :param it: number of iterations
    :return:
    """
    objective = create_objective_function(K, L, init, c)
    gradient = create_gradient_function(K, L, init, c)
    pi, mu_est = cvm.monotone_fista_support(objective, gradient, init, mu, mu_min, it, proj.neg_projector)
    return pi
