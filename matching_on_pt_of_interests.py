import os.path as op
import pickle
from convex_multi_hsic import *
import networkx as nx
from show_results_on_sphere import *
from util import *


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
    for subject in subjects_list:
        gram_path = op.join(gram_dir, 'K_{}_{}.pck'.format(subject, hem))
        graph_path = graph_dir + 'full_lh_' + subjects_list[nb_s] + '_pitgraph.gpickle'
        f = open(gram_path, 'rb')
        K = pickle.load(f)
        center_K = centered_matrix(K[:, :, subkernel_ind])  # centrage des matrices
        center_K = normalized_matrix(center_K)   # normalisation des données
        K_list.append(center_K)
        G = nx.read_gpickle(graph_path)
        graph_list.append(G)
        nb_s += 1

    pt_interest1 = [-43, 6, 89]
    pt_interest2 = [-48, 8, -87]

    list_pits_to_remove = list()
    rayon_limite = 80

    for g in range(len(graph_list)):
        for p in range(graph_list[g].number_of_nodes()):
            xp = graph_list[g].nodes[p]['coord'][0]
            yp = graph_list[g].nodes[p]['coord'][1]
            zp = graph_list[g].nodes[p]['coord'][2]
            new = list()
            list_pits_to_remove.append(new)
            if dist_on_sphere(100, -43, 6, 89, xp, yp, zp) > rayon_limite:
            # if dist_on_sphere(100, -48, 8, -87, xp, yp, zp) > 60:
                list_pits_to_remove[g].append(p)

    for k in range(len(K_list)):
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=0)
        K_list[k] = np.delete(K_list[k], list_pits_to_remove[k], axis=1)
        for node in list_pits_to_remove[k]:
            graph_list[k].remove_node(node)

    pits_sphere(graph_list)