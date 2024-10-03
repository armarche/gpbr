
import numpy as np
import matplotlib.pyplot as plt
from mfs.collocation import collocation_2d, source_points_2d
from mfs.boundary import starlike_circle, starlike_curve


## Plot
def plot_2d(g1_mesh, g2_mesh, inner_art_boundary_mesh, outter_art_boundary_mesh):
    fig_plot, ax_plot = plt.subplots()
    ax_plot.plot(g1_mesh[0], g1_mesh[1], 'b--')
    ax_plot.plot(g2_mesh[0], g2_mesh[1], 'r--')
    ax_plot.plot(inner_art_boundary_mesh[0], inner_art_boundary_mesh[1], 'ro')
    ax_plot.plot(outter_art_boundary_mesh[0], outter_art_boundary_mesh[1], 'bo')

    ax_plot.axis('equal')
    plt.grid()
    plt.show()
    plt.close()




if __name__ == '__main__':
    n = 16
    collocation_points = collocation_2d(n)
    collocation_points_closed = np.insert(collocation_points, 0, 0)

    source_points = source_points_2d(n)
    # source_points_closed = np.insert(source_points, 0, 0, axis=1)

    circle_points_mesh_closed = starlike_circle(collocation_points_closed)
    circle_points_mesh = np.delete(circle_points_mesh_closed, 0, axis=0)
    

    g1_rvalues = np.ones(len(collocation_points_closed))
    g2_rvalues = np.ones(len(collocation_points_closed))*2

    g1_curve_mesh_closed = starlike_curve(g1_rvalues, circle_points_mesh_closed)
    g2_curve_mesh_closed = starlike_curve(g2_rvalues, circle_points_mesh_closed)

 
    # print(curve_points_closed)
    eta1 = 0.5
    eta2 = 2
    g1_art_values = eta1*g1_curve_mesh_closed
    g2_art_values = eta2*g2_curve_mesh_closed

    # art_g1_curve_closed = starlike_curve(eta1*g1_rvalues, circle_points_closed)
    # art_g2_curve_closed = starlike_curve(eta2*g2_rvalues, circle_points_closed)

    plot_2d(g1_curve_mesh_closed, g2_curve_mesh_closed, g1_art_values, g2_art_values)