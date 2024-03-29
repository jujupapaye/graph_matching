# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:31:44 2012

@author: Sylvain Takerkart
"""

import numpy as np
import networkx as nx

from sklearn.metrics.pairwise import rbf_kernel

def vprod(x):
    y = np.atleast_2d(x)
    return np.dot(y.T, y)


class sxdnewkernel:
    """Our graph kernel that combines  structural information (s), geometrical
    information (coordinates x) and attributes information (depth d)
    In this "new" version, add one to the diagonal of the adjacency matrices to
    make sure the kernel cannot return 0.
    """

    def __init__(self, x_sigma=1.0, d_sigma=1.0, subkernels=False):
        """Initialize the kernel
            x_sigma : bandwidth used for geometrical attributes (coordinates)
            d_sigma : bandwidth used for the depth attribute
        """

        self.d_sigma = d_sigma
        self.x_sigma = x_sigma

        # return all subkernels or not?
        # default = return only the full sga kernel
        self.subkernels = subkernels

    def evaluate(self, g1, g2):
        """Evaluate the kernel value for two input graphs g1 and g2
        """

        # extract the vectors and matrices necessary to compute the kernel from the networkx graphs
        coords_dict = nx.get_node_attributes(g1, 'coord')
        X1 = np.array([coords_dict[i] for i in g1.nodes()])
        coords_dict = nx.get_node_attributes(g2, 'coord')
        X2 = np.array([coords_dict[i] for i in g2.nodes()])
        depth_dict = nx.get_node_attributes(g1, 'depth')
        D1 = np.atleast_2d(np.array([depth_dict[i] for i in g1.nodes()])).T
        depth_dict = nx.get_node_attributes(g2, 'depth')
        D2 = np.atleast_2d(np.array([depth_dict[i] for i in g2.nodes()])).T
        A1 = nx.to_numpy_matrix(g1)
        A2 = nx.to_numpy_matrix(g2)

        # depth contribution using gaussian kernel
        K_dd = rbf_kernel(D1,D2,gamma = 0.5 / (self.d_sigma**2)).flatten()
        K_d = vprod(K_dd)  # matrix product of the two vectors K_dd.T and K_dd

        # geometrical (coordinates) contribution using gaussian kernel
        K_xx = rbf_kernel(X1,X2,gamma = 0.5 / (self.x_sigma**2)).flatten()
        K_x = vprod(K_xx)

        # structural contribution using linear kernel on binary values
        direct_adjacency = np.kron(A1, A2)
        K_s = direct_adjacency

        # with two of the three components
        K_xd = np.multiply(K_x, K_d)
        K_sd = np.multiply(K_s, K_d)
        K_sx = np.multiply(K_s, K_x)
        # the full kernel with the three components
        K_sxd = np.multiply(K_sx, K_d)

        # sum all terms of these babies to compute the kernel values
        K_list = [K_sxd.sum(), K_xd.sum(), K_sd.sum(), K_sx.sum(), K_s.sum(),
                  K_x.sum(), K_d.sum()]

        if self.subkernels:
            # return the full K_sga kernel and other kernels which do not use
            # all three types of features
            return np.array(K_list)
        else:
            # return only the full K_sga kernel
            return K_sxd.sum()

