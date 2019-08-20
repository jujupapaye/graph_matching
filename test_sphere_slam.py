
import slam.generate_parametric_surfaces as sps
import slam.plot as splt

if __name__ == '__main__':
    sphere_mesh = sps.generate_sphere(100)
    z_coord_texture = sphere_mesh.vertices[:, 2]
    splt.pyglet_plot(sphere_mesh, z_coord_texture, caption="Sphere",color_map='Reds')