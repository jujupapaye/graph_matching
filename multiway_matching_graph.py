"""
Tests du matching de plusieurs graphes de pits
"""

import os.path as op
import pickle
from convex_multi_hsic import *
import networkx as nx
from show_results_on_sphere import *
from util import *

if __name__ == '__main__':
    # parameters
    hem = 'lh'
    graph_type = 'radius'
    graph_param = 60

    # define directories
    gram_dir = '/home/jul/Documents/cours/STAGE/CKS/radius_60/'
    graph_dir = '/home/jul/Documents/cours/STAGE/CKS/pytorch_graph_pits/data/'

    subjects_list = ['OAS1_0006', 'OAS1_0009', 'OAS1_0025', 'OAS1_0049', 'OAS1_0051', 'OAS1_0054', 'OAS1_0055',
                     'OAS1_0057', 'OAS1_0059', 'OAS1_0061', 'OAS1_0077', 'OAS1_0079', 'OAS1_0080', 'OAS1_0087',
                     'OAS1_0104', 'OAS1_0125', 'OAS1_0136', 'OAS1_0147', 'OAS1_0150', 'OAS1_0151', 'OAS1_0152',
                     'OAS1_0156', 'OAS1_0162', 'OAS1_0191', 'OAS1_0192', 'OAS1_0193', 'OAS1_0202', 'OAS1_0209',
                     'OAS1_0218', 'OAS1_0224', 'OAS1_0227', 'OAS1_0231', 'OAS1_0236', 'OAS1_0239', 'OAS1_0246',
                     'OAS1_0249', 'OAS1_0253', 'OAS1_0258', 'OAS1_0294', 'OAS1_0295', 'OAS1_0296', 'OAS1_0310',
                     'OAS1_0311', 'OAS1_0313', 'OAS1_0325', 'OAS1_0348', 'OAS1_0379', 'OAS1_0386', 'OAS1_0387',
                     'OAS1_0392', 'OAS1_0394', 'OAS1_0395', 'OAS1_0397', 'OAS1_0406', 'OAS1_0408', 'OAS1_0410',
                     'OAS1_0413', 'OAS1_0415', 'OAS1_0416', 'OAS1_0417', 'OAS1_0419', 'OAS1_0420', 'OAS1_0421',
                     'OAS1_0431', 'OAS1_0437', 'OAS1_0442', 'OAS1_0448', 'OAS1_0004', 'OAS1_0005', 'OAS1_0007',
                     'OAS1_0012', 'OAS1_0017', 'OAS1_0029', 'OAS1_0037', 'OAS1_0043', 'OAS1_0045', 'OAS1_0069',
                     'OAS1_0090', 'OAS1_0092', 'OAS1_0095', 'OAS1_0097', 'OAS1_0101', 'OAS1_0102', 'OAS1_0105',
                     'OAS1_0107', 'OAS1_0108', 'OAS1_0111', 'OAS1_0117', 'OAS1_0119', 'OAS1_0121', 'OAS1_0126',
                     'OAS1_0127', 'OAS1_0131', 'OAS1_0132', 'OAS1_0141', 'OAS1_0144', 'OAS1_0145', 'OAS1_0148',
                     'OAS1_0153', 'OAS1_0174', 'OAS1_0189', 'OAS1_0211', 'OAS1_0214', 'OAS1_0232', 'OAS1_0250',
                     'OAS1_0261', 'OAS1_0264', 'OAS1_0277', 'OAS1_0281', 'OAS1_0285', 'OAS1_0302', 'OAS1_0314',
                     'OAS1_0318', 'OAS1_0319', 'OAS1_0321', 'OAS1_0328', 'OAS1_0333', 'OAS1_0340', 'OAS1_0344',
                     'OAS1_0346', 'OAS1_0350', 'OAS1_0359', 'OAS1_0361', 'OAS1_0368', 'OAS1_0370', 'OAS1_0376',
                     'OAS1_0377', 'OAS1_0385', 'OAS1_0396', 'OAS1_0403', 'OAS1_0409', 'OAS1_0435', 'OAS1_0439',
                     'OAS1_0450']

    # choice of which kernel we will be using to perform the matching...
    # subkernel_ind = 0 # using the full info: structure, coordinates, depth
    # subkernel_ind = 1 # using coordinates, depth
    # subkernel_ind = 2 # using structure, depth
    # subkernel_ind = 3 # using structure, coordinates
    # subkernel_ind = 4 # using structure
    subkernel_ind = 5  # using coordinates
    # subkernel_ind = 6 # using depth
    # the relevant choices are:
    # 5 (this should give us a baseline)
    # 3 (this should improve things a bit by adding the structure of the graph in the kernel computation)
    # 0 (hopefully adding the depth should help a bit more)

    K_list = []  # list of gram matrix
    graph_list = []  # list of corresponding graphs
    nb_s = 0  # number of subjects
    number = 0
    for subject in subjects_list:
        gram_path = op.join(gram_dir, 'K_{}_{}.pck'.format(subject, hem))
        graph_path = graph_dir + 'full_lh_' + subjects_list[nb_s] + '_pitgraph.gpickle'
        f = open(gram_path, 'rb')
        K = pickle.load(f)
        if K.shape[0] == 86 and number <= 3:
            number += 1
            center_K = centered_matrix(K[:, :, subkernel_ind])  # centrage des matrices
            center_K = normalized_matrix(center_K)   # normalisation des données
            K_list.append(center_K)
            G = nx.read_gpickle(graph_path)
            graph_list.append(G)
        nb_s += 1


    # traitement des matrices de gram -> les mettre toute à la même taille

    nb_patients = len(K_list)
    nb_pits = 86
    # new_K_list = np.zeros((nb_patients, max, max))
    # new_K_list.fill(np.nan)

    '''for k in range(len(K_list)):
        for i in range(K_list[k].shape[0]):
            for j in range(K_list[k].shape[1]):
                new_K_list[i, j] = K_list[k][i, j]'''

    perms = np.zeros((nb_patients, nb_pits, nb_pits))
    perms[0] = np.eye(nb_pits)

    for p in range(1, len(perms)):
        perms[p] = init_eig(K_list[0], K_list[p], nb_pits)
        # perms[p] = init_random(nb_pits)   # initialisation des permutations

    # parameters of the gradient descent
    c, mu, mu_min, it, nb_tour = 1e12, 1e-3, 1e-20, 1000, 3    # (params pour 3 patients)
    # c, mu, mu_min, it, nb_tour = 1, 1e-10, 1e-30, 500, 3

    init = perms.copy()
    perms_opt = estimate_perms(K_list, perms, c, mu, mu_min, it, nb_tour)

    t = np.zeros((nb_patients, nb_pits, 2))
    res = np.zeros((nb_patients, nb_pits, nb_pits))
    perms_opt = perms
    for i in range(1, nb_patients):
        res[i], t[i] = transformation_permutation(perms_opt[i])
    show_sphere(t, graph_list)
