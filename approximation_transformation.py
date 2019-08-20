######################

#Fonctions qui transforme une matrice p
#en matrice de permutation de différentes manières

######################



import numpy as np
import convexminimization2 as cvm
#from munkres import Munkres
from projector import *


def create_objective_function_approx(M,c):
    def objective_function(beta):
        n = M.shape[0]
        UN = np.ones((n, 1))
        constraints = np.linalg.norm(beta @ UN - UN)**2 + np.linalg.norm(beta.T @ UN - UN)**2
        result = 0.5*(np.linalg.norm(beta - M))**2 + c*0.5*constraints
        return result
    return objective_function

#creation du gradient ou M est la matrice à approximer
def create_gradient_function_approx(M,c):
    def gradient_function(beta):
        n = M.shape[0]
        g = beta - M
        g += c*(beta.T + beta - 2*np.eye(n)) @ np.ones((n, n))
        return g
    return gradient_function


def estimate_permutation_opt(M, c, mu, mu_min, it, F):
    objective = create_objective_function_approx(M, c)
    gradient = create_gradient_function_approx(M, c)
    projector = create_projector_positivite_fixation(F, M)
    (res, mu) = cvm.monotone_fista_support(objective, gradient, M, mu,mu_min, it, projector)
    return res

#Transforme une matrice avec des floats en une matrice de permutation en cherchant les max et en mettant à zero les lignes / colonnes
def transformation_permutation_opt(perm):
    p = perm.copy()
    n = p.shape[0]
    filtre = np.ones((n, n))
    result = np.ones((n, 2), int)
    for i in range(n):
        new = p * filtre
        max = new.max()
        argmax = new.argmax()
        c = argmax % n      #colonne du max
        l = int(argmax/n)  #ligne du max
        p[l, :] = 0        #transformation de la permutation
        p[:, c] = 0
        p[l, c] = 1
        filtre[l, :] = 0  #mise à jour du filtre
        filtre[:, c] = 0
        result[i][0] = l
        result[i][1] = c
        p = estimate_permutation_opt(p, 1, 0.1, 0.0000001, 80, filtre)
    return p, result

#Transforme une matrice avec des floats en une matrice de permutation en cherchant les max et en mettant à zero les lignes / colonnes
def transformation_permutation(p):
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
            if new[j, c]!=-1:
                new[j, c] = 0
            if new[l, j] != -1:
                new[l, j] = 0
    return (-1) * new, result


#transformation d'une matrice p à une permutation grace à l'algorithme hongrois
'''def transformation_permutation_hungarian(p):
    m = Munkres()
    new = -1*p
    indexes = m.compute(new)
    res = np.zeros((p.shape[0], p.shape[0]))
    t = np.zeros((p.shape[0], 2))
    ind = 0
    for i, j in indexes:
        res[i, j]=1
        t[ind, 0] = i
        t[ind, 1] = j
        ind += 1
    return res, t'''
