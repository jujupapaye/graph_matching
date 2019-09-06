import slam.generate_parametric_surfaces as sps
import slam.plot as splt
import numpy as np
import trimesh as trim

if __name__ == '__main__':
    sphere_mesh = sps.generate_sphere(1000)
    sphere_mesh.vertices = 10*sphere_mesh.vertices
    z_coord_texture = sphere_mesh.vertices[:, 0]*100
    #z_coord_texture = np.ones(sphere_mesh.vertices.shape[0]) * 200
    splt.pyglet_plot(sphere_mesh, z_coord_texture, caption="Sphere", color_map='Reds')

