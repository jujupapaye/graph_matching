import numpy as np


def centered_matrix(K):
    """
    Centre une matrice carré K
    :param K: matrice
    :return: HKH où H=(I-1)/n  1=matrice de 1 de taille nxn
    """
    n = K.shape[0]
    unit = np.ones((n, n))  # matrice de 1 n*n
    I = np.eye(n)  # matrice identité n*n
    H = I-unit/n
    return np.dot(np.dot(H, K), H)


def normalized_matrix(K):
    """
    Normalise une matrice
    :param K: matrice
    :return: matrice K normalisé
    """
    new = np.zeros((K.shape[0], K.shape[1]))
    for i in range(K.shape[0]):
        for j in range(i, K.shape[1]):
            new[i, j] = K[i, j] / np.sqrt(K[i, i] * K[j, j])
    new = (new + new.T)/2  # symetrie
    return new


def init_random(n):
    """
    initialisation de matrice de permutation aléatoire
    :param n_obs: taille de la matrice carré voulu
    :return: permutation aléatoire n*n
    """
    bases = np.eye(n)
    init = np.random.permutation(n)
    PI_0 = bases[init, :]
    return PI_0


def init_eig(K, L, n_obs):
    """
    initialisation avec les 'eigen vectors' triés
    """
    [U_K, V_K] = np.linalg.eig(K)
    [U_L, V_L] = np.linalg.eig(L)
    i_VK = np.argsort(-V_K[:, 0])
    i_VL = np.argsort(-V_L[:, 0])
    PI_0 = np.zeros((n_obs, n_obs))
    PI_0[np.array(i_VL), np.array(i_VK)] = 1
    return PI_0


def dist_on_sphere(r, xA, yA, zA, xB, yB, zB):
    """
    Calcule la distance de 2 points A(xA,yA,zA) et B(xB, yB, zB) sur une sphere de rayon r
    :return: distance géodésique entre A et B
    """
    return r * 2 * np.arcsin(dist_eucli(xA, yA, zA, xB, yB, zB)/(2*r))


def dist_eucli(xA, yA, zA, xB, yB, zB):
    """
    Calcule la distance euclidienne entre 2 points A(xA,yA,zA) et B(xB, yB, zB)
    :return: distance euclidienne entre A et B
    """
    return np.sqrt((xB-xA)**2 + (yB-yA)**2 + (zB-zA)**2)
