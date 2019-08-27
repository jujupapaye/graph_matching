import matplotlib.pyplot as plt
import numpy as np
import trimesh as trim
import slam.plot as splt


def show_sphere_for_2_patients(matching, g0, g1):
    """
    (not useful)
    Representation of two patients matching of sulcal pits
    Show a sphere representing a brain where points are sulcal pits. Two points have the same colors if they are matched
    (using matplotlib)

    Parameters :
    -matching : array of sorted indices where the node number matching[i] of g0 correspond to the i node of g1
    -g0, g1: graphs to match

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    ax.set_facecolor('black')

    colormap = np.array(['black', 'darkred', 'sienna', 'darkcyan', 'pink', 'r', 'g', 'b', 'midnightblue', 'violet',
                         'gold'])
    markers = np.array(['.', "o", "v", "s", "P", "*", "x"])

    m = 0  # numero du markers

    for node in range(matching.shape[0]):
        if node % colormap.shape[0]:
            m = (m + 1) % markers.shape[0]
        x0 = g0.nodes[matching[node]]['coord'][0]
        y0 = g0.nodes[matching[node]]['coord'][1]
        z0 = g0.nodes[matching[node]]['coord'][2]
        x1 = g1.nodes[node]['coord'][0]
        y1 = g1.nodes[node]['coord'][1]
        z1 = g1.nodes[node]['coord'][2]
        ax.scatter(x0, y0, z0, c=colormap[node % colormap.shape[0]], marker=markers[m])
        ax.scatter(x1, y1, z1, c=colormap[node % colormap.shape[0]], marker=markers[m])

    plt.show()


def show_sphere_for_2(matching, g0, g1):
    """
    Representation of two patients matching of sulcal pits
    Show a sphere representing a brain where points are sulcal pits. Two points have the same colors if they are matched
    (using trimesh/ slam)

    Parameters :
    -matching : array where the node number matching[i] of g0 correspond to the i node of g1
    -g0, g1: graphs to match

    """
    color_hex = np.array(['#000000', '#6e2c00',  '#7d6608', '#186a3b', '#1b4f72', '#4a235a', '#641e16', '#ecf0f1',
                          '#ecf0f1', '#f1c40f', '#27ae60', '#48c9b0', '#2980b9', '#cb4335', '#af7ac5', '#a3e4d7', '#fad7a0', '#f2d7d5'])
    sphere_mesh = trim.primitives.Sphere(radius=100)
    transfo_pit0 = np.eye(4)
    transfo_pit1 = np.eye(4)
    pairs_nb = 0
    nb_pits = matching.shape[0]

    pit_col_val = np.arange(0, nb_pits, 1)
    textures = np.ones(sphere_mesh.vertices.shape[0]) * nb_pits

    for node in range(0, matching.shape[0]):
        pairs_nb += 1
        color = trim.visual.color.hex_to_rgba(color_hex[node % color_hex.shape[0]])

        if g0.has_node(matching[node]) and g1.has_node(node):  # si les noeuds matchés existe bien dans les 2 graphes
            x0 = g0.nodes[matching[node]]['coord'][0]
            y0 = g0.nodes[matching[node]]['coord'][1]
            z0 = g0.nodes[matching[node]]['coord'][2]
            pit0 = trim.primitives.Sphere(radius=1, subdivisions=0)
            transfo_pit0[:, 3] = [x0, y0, z0, 1]
            pit0.apply_transform(transfo_pit0)
            pit0.visual.face_colors = color
            textures = np.concatenate((textures, np.ones(pit0.vertices.shape[0]) * pit_col_val[node]))

            x1 = g1.nodes[node]['coord'][0]
            y1 = g1.nodes[node]['coord'][1]
            z1 = g1.nodes[node]['coord'][2]
            pit1 = trim.primitives.Sphere(radius=1, subdivisions=0)
            transfo_pit1[:, 3] = [x1, y1, z1, 1]
            pit1.apply_transform(transfo_pit1)
            pit1.visual.face_colors = color
            textures = np.concatenate((textures, np.ones(pit1.vertices.shape[0]) * pit_col_val[node]))

            sphere_mesh = sphere_mesh + pit0 + pit1

    splt.pyglet_plot(sphere_mesh, values=textures, color_map='gist_ncar', plot_colormap=True)


def show_sphere(matching_list, graphs):
    """
    Representation of multiple graphs matching of sulcal pits
    Show a sphere representing a brain where points are sulcal pits. Two points have the same colors if they are matched
    (using trimesh/ slam)

    :param matching_list: list of array of matching indices
    :param graphs: list of corresponding graphs
    """
    sphere_mesh = trim.primitives.Sphere(radius=100)

    nb_patients = matching_list.shape[0]
    nb_pits = matching_list[0].shape[0]

    pit_col_val = np.arange(0, nb_pits, 1)
    textures = np.ones(sphere_mesh.vertices.shape[0])*nb_pits

    # tri des résultats pour faciliter l'interpretation après
    for p in range(nb_patients):
        matching_list[p] = matching_list[p][np.argsort(matching_list[p][:, 1])]

    trans_pit0 = np.eye(4)
    trans_pit1 = np.eye(4)

    for pit in range(0, nb_pits):
        x0 = graphs[0].nodes[pit]['coord'][0]
        y0 = graphs[0].nodes[pit]['coord'][1]
        z0 = graphs[0].nodes[pit]['coord'][2]
        pit0 = trim.primitives.Sphere(radius=1, subdivisions=0)
        trans_pit0[:, 3] = [x0, y0, z0, 1]
        pit0.apply_transform(trans_pit0)
        textures = np.concatenate((textures, np.ones(pit0.vertices.shape[0]) * pit_col_val[pit]))
        sphere_mesh += pit0

        for pat in range(nb_patients):

            x1 = graphs[pat].nodes[matching_list[pat][pit, 0]]['coord'][0]
            y1 = graphs[pat].nodes[matching_list[pat][pit, 0]]['coord'][1]
            z1 = graphs[pat].nodes[matching_list[pat][pit, 0]]['coord'][2]
            pit1 = trim.primitives.Sphere(radius=1, subdivisions=0)
            trans_pit1[:, 3] = [x1, y1, z1, 1]
            pit1.apply_transform(trans_pit1)
            textures = np.concatenate((textures, np.ones(pit1.vertices.shape[0]) * pit_col_val[pit]))

            sphere_mesh += pit1

    splt.pyglet_plot(sphere_mesh, values=textures, color_map='gist_ncar', plot_colormap=True, caption='test')


def pits_sphere(graphs):
    """
    Representation of pits graphs on a sphere
    Show a sphere representing a brain where points are sulcal pits. One color for one patient

    :param graphs: list of pits graphs
    """
    color_hex = np.array(['#000000', '#6e2c00', '#7d6608', '#186a3b', '#1b4f72', '#4a235a', '#641e16', '#ecf0f1',
                          '#ecf0f1', '#f1c40f', '#27ae60', '#48c9b0', '#2980b9', '#cb4335', '#af7ac5', '#a3e4d7', '#fad7a0', '#f2d7d5'])
    sphere_mesh = trim.primitives.Sphere(radius=100)

    trans_pit = np.eye(4)
    nb_p = 0
    for g in range(len(graphs)):
        color = trim.visual.color.hex_to_rgba(color_hex[g % color_hex.shape[0]])
        for p in graphs[g].nodes:
            x = graphs[g].nodes[p]['coord'][0]
            y = graphs[g].nodes[p]['coord'][1]
            z = graphs[g].nodes[p]['coord'][2]
            pit = trim.primitives.Sphere(radius=1, subdivisions=0)
            trans_pit[:, 3] = [x, y, z, 1]
            pit.apply_transform(trans_pit)
            pit.visual.face_colors = color
            sphere_mesh += pit
            nb_p += 1
        if nb_p == color_hex.shape[0] - 1:
            break

    sphere_mesh.show()
