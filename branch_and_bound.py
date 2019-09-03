"""

 Algo branch and bround pour la comparaison de 2 ensemble
 (changer les fonctions objectif/gradient dans le calcul de la lower bounds
 pour changer l'utilité de l'algo)

"""

import numpy as np
import convex_simple_hsic as hsic
import approximation_transformation as transfo
import projector as proj
import convexminimization2 as cvm
from numba import jit


def compute_lower_bound(K, L, init, c, mu, mu_min, it, constraint):
    """
    Estime la borne inférieur
    :param K: premiere matrice de gram
    :param L: seconde matrice de gram
    :param init: initialisation pour la descente du gradient pour l'estimation de la borne inf
    :param c: hyperparamètre pour la descente du gradient pour l'estimation de la borne inf
    :param mu: mu pour la descente du gradient pour l'estimation de la borne inf
    :param mu_min: mu_min pour la descente du gradient pour l'estimation de la borne inf
    :param it: nombre d'iterations dans la descente du gradient pour l'estimation de la borne inf
    :param constraint: les contraintes fixés qu'on ne peut pas toucher
    :return: estimation de la borne inférieur avec ses contraintes
    """
    objective = hsic.create_objective_function(K, L, init, c)
    gradient = hsic.create_gradient_function(K, L, init, c)
    projector = proj.create_projector_neg_and_fixed(constraint)
    res, mu = cvm.monotone_fista_support(objective, gradient, init, mu, mu_min, it, projector)
    filtre = proj.creation_filtre(constraint, K.shape[0])
    i0, j0 = np.unravel_index(np.argmax(res*filtre, axis=None), res.shape)  # i0,j0 les indices de l'élément max sans les valeurs fixées
    return res, objective(res), i0, j0


def compute_upper_bound(res_lower, c, K, L):
    """
    Calcule la borne inférieur
    :param res_lower: matrice stochastique (comportant réels entre 0 et 1 et somme colonne/ligne à 1)
    :param c: hyperparametre
    :param K: matrice de Gram
    :param L: matrice de Gram
    :return: borne inférieur
    """
    p = transfo.transformation_permutation_hungarian(res_lower)[0]
    UN = np.ones((p.shape[0], 1))
    constraints = np.linalg.norm(p @ UN - UN)**2 + np.linalg.norm(p.T @ UN - UN)**2
    return np.linalg.norm(K @ p.T - p.T @ L.T)**2 + c*constraints


def choose_next_contraint(act_constraint, lowers):
    """
    Heuristique pour choisir la prochaine contrainte à choisir
    :param act_constraint: liste des contraintes actives
    :param lowers: liste de leurs bornes inférieurs associés
    :return: contrainte à choisir, sa borne inférieur
    """
    minimum = np.argmin(lowers)  # choix de la contrainte ayant la plus petite borne inf
    return act_constraint[minimum], minimum


@jit(parallel=True)
def branch_and_bound(K1, K2, init, c, mu, mu_min, it):
    """
    Algorithme du branch and bound pour l'appariement de 2 ensembles (min || ||+c*contraintes)
    :param K1: matrice de gram du premier ensemble à comparer
    :param K2: matrice de gram du second ensemble à comparer
    :param init: matrice d'initialisation
    :param c: hyperparamètre pour la borne inf (minimisation convexe)
    :param mu: mu pour la borne inf (minimisation convexe)
    :param mu_min: mu_min pour la borne inf (minimisation convexe)
    :param it: nombre d'itération pour trouver la lower bound
    :return: une liste d'une seule contrainte (la seule restante) qui est la solution optimale
    """
    n = K1.shape[0]
    lowers = list()   # list of lowers bounds
    uppers = list()   # list of uppers bounds
    active_constraints = list()   # active constraints
    nb_cons = 0
    for i in range(n):
        constraint = list()
        constraint.append((0, i))
        active_constraints.append(constraint)  # initialisation of active_constraints
        p, l, i0, j0 = compute_lower_bound(K1, K2, init.copy(), c, mu, mu_min, it, constraint)
        lowers.append(l)   # initialisation of lowers and uppers
        u = compute_upper_bound(p, c, K1, K2)
        uppers.append(u)
    L = min(lowers)  # initialisation of L and U
    U = min(uppers)
    while L != U or len(active_constraints) != 1:
        chosen_constraint, ind = choose_next_contraint(active_constraints, lowers)  # choix de la contrainte avec la + petite borne inférieur
        nb_cons += 1
        p, l, i0, j0 = compute_lower_bound(K1, K2, init.copy(), c, mu, mu_min, it, chosen_constraint)
        r = proj.perm_with_constraint(chosen_constraint, n)
        upper_bound_chosen_constraint = uppers[ind]
        for j in range(n):
            if np.sum(r[:, j]) == 0 and np.sum(r[i0, :] == 0):  # si cette contrainte est "faisable"
                new_c = chosen_constraint.copy()
                new_c.append((i0, j))  # creation d'une nouvelle contrainte
                active_constraints.append(new_c)
                p, l, ii, jj = compute_lower_bound(K1, K2, init.copy(), c, mu, mu_min, it, new_c)
                u = compute_upper_bound(p, c, K1, K2)
                lowers.append(l)
                uppers.append(u)
                if u > upper_bound_chosen_constraint:
                    uppers[ind] = u
                    upper_bound_chosen_constraint = uppers[ind]
        del active_constraints[ind]
        del lowers[ind]
        del uppers[ind]
        ind_uppers = np.argmin(uppers)
        L = min(lowers)  # mise à jour de L et U
        U = min(uppers)
        j = len(lowers)-1
        while j >= 0:  # on supprime les contraintes c tel que lower_bound(c) > U
            if lowers[j] > U and ind_uppers != j:
                del active_constraints[j]
                del lowers[j]
                del uppers[j]
            j = j-1
    print("Nombres de contraintes (chemins) étudiées :", nb_cons, "/", np.math.factorial(n))
    return active_constraints
