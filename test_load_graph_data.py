import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np

if __name__ == "__main__":
    dir = '/home/jul/Documents/cours/STAGE/CKS/pytorch_graph_pits/data/'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #plot the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 99 * np.outer(np.cos(u), np.sin(v))
    y = 99 * np.outer(np.sin(u), np.sin(v))
    z = 99 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='white')

    colormap = np.array(['black', 'grey', 'darkred', 'darkcyan', 'r', 'g', 'b', 'midnightblue', 'violet'])
    count = 0
    for graph_path in os.listdir(dir):
        G = nx.read_gpickle(dir+graph_path)
        count += 1
        if count < 10:
            for node in G.nodes:
                if node < colormap.shape[0]:
                    x = G.nodes[node]['coord'][0]
                    y = G.nodes[node]['coord'][1]
                    z = G.nodes[node]['coord'][2]
                    ax.scatter(x, y, z, c=colormap[node % colormap.shape[0]], marker='.')
    plt.show()
