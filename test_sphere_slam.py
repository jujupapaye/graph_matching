import slam.generate_parametric_surfaces as sps
import slam.plot as splt
import numpy as np
import trimesh as trim

if __name__ == '__main__':
    '''sphere_mesh = sps.generate_sphere(1000)
    sphere_mesh.vertices = 10*sphere_mesh.vertices
    z_coord_texture = sphere_mesh.vertices[:, 2]
    #splt.pyglet_plot(sphere_mesh, z_coord_texture, caption="Sphere", color_map='Reds')
    transfo_pit = np.eye(4)
    transfo_pit[:, 3] = [10, 0, 0, 1]
    sphere_mesh2 = sps.generate_sphere(100)
    sphere_mesh2.vertices = 1 * sphere_mesh2.vertices
    sphere_mesh2.apply_transform(transfo_pit)
    #splt.pyglet_plot(sphere_mesh, z_coord_texture, caption="Sphere", color_map='Blues')
    sphere_mesh += sphere_mesh2
    splt.pyglet_plot(sphere_mesh)'''


    s = trim.primitives.Sphere()
    s.show()

    s = trim.primitives.Sphere(subdivisions=1)
    s.show()
