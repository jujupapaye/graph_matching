import networkx as nx
import os
import numpy as np
import trimesh as trim
import slam.io as sio
import nibabel as nb
import slam.plot as splt

if __name__ == "__main__":
    dir = '/home/jul/Documents/cours/STAGE/CKS/pytorch_graph_pits/data/'
    data_path = '/home/jul/Documents/cours/STAGE/CKS/graph_matching/data'
    subj_list = list()
    graphs = list()
    files_list = os.listdir(dir)
    subject = 'OAS1_0004'
    for graph_path in files_list:
        if subject in graph_path and 'lh' in graph_path:
            G = nx.read_gpickle(os.path.join(dir, graph_path))
            graphs.append(G)
            subj_list.append(graph_path[8:17])
    print(subj_list)

    trans_pit0 = np.eye(4)
    trans_pit1 = np.eye(4)

    subj_ind = 0
    meshes = sio.load_mesh(os.path.join(data_path, subj_list[subj_ind]+'.lh.sphere.reg.gii'))
    parcels = nb.freesurfer.read_annot(os.path.join(data_path, subj_list[subj_ind]+'.lh.aparc.annot'))
    parcels_value = parcels[0]
    textures = parcels_value

    nb_pits = len(graphs[subj_ind].nodes)
    # pit_col_val = 10
    pit_col_val = np.arange(10, (nb_pits+1)*40, 40)
    for pit in range(nb_pits):
        x0 = graphs[subj_ind].nodes[pit]['coord'][0]
        y0 = graphs[subj_ind].nodes[pit]['coord'][1]
        z0 = graphs[subj_ind].nodes[pit]['coord'][2]
        pit0 = trim.primitives.Sphere(radius=1, subdivisions=0)
        trans_pit0[:, 3] = [x0, y0, z0, 1]
        pit0.apply_transform(trans_pit0)

        meshes += pit0
        textures = np.concatenate((textures, np.ones(pit0.vertices.shape[0]) * pit_col_val[pit]))



    ROI1 = trim.primitives.Sphere(radius=80, subdivisions=1)
    trans_pit0[:, 3] = [-43, 6, 89, 1]
    ROI1.apply_transform(trans_pit0)
    meshes += ROI1
    textures = np.concatenate((textures, np.ones(ROI1.vertices.shape[0]) * 20))
    ROI2 = trim.primitives.Sphere(radius=60, subdivisions=1)
    trans_pit0[:, 3] = [-48, 8, -87, 1]
    ROI2.apply_transform(trans_pit0)
    meshes += ROI2
    textures = np.concatenate((textures, np.ones(ROI2.vertices.shape[0]) * 40))
    splt.pyglet_plot(meshes, values=textures, color_map='jet', plot_colormap=True, caption='test')



    white_mesh = sio.load_mesh(os.path.join(data_path, subj_list[subj_ind]+'.lh.white.gii'))
    splt.pyglet_plot(white_mesh, values=parcels_value, color_map='jet', plot_colormap=True, caption='test')



