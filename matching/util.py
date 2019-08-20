import numpy as np

#centre une matrice ,retourne HKH où H=(I-1)/n  1=matrice de 1 de taille nxn
def centered_matrix(K):
    n=K.shape[0]
    unit=np.ones((n, n))  #matrice de 1 n*n
    I=np.eye(n)  #matrice identité n*n
    H=I-unit/n
    return np.dot(np.dot(H, K), H)

#normalise une matrice
def normalized_matrix(K):
    new=np.zeros((K.shape[0],K.shape[1]))
    for i in range(K.shape[0]):
        for j in range(i,K.shape[1]):
            new[i,j] = K[i,j] / np.sqrt(K[i,i] * K[j,j])
    new = (new + new.T)/2 #symetrie
    return new

# initialisation aléatoire
def init_random(n_obs):
    bases = np.eye(n_obs)
    init = np.random.permutation(n_obs)
    PI_0 = bases[init,:]
    return PI_0

# initialisation avec les 'eigen vectors' triés
def init_eig(K,L,n_obs):
    [U_K,V_K] = np.linalg.eig(K)
    [U_L,V_L] = np.linalg.eig(L)
    i_VK = np.argsort(-V_K[:,0])
    i_VL = np.argsort(-V_L[:,0])
    PI_0 = np.zeros((n_obs,n_obs))
    PI_0[np.array(i_VL),np.array(i_VK)] = 1
    return PI_0
