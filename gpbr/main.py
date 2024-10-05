from gpbr.collocation import collocation_points_2d, CollocationData2D
from gpbr.boundary import StarlikeCurve, starlike_circle_base, starlike_curve
import cloudpickle

import numpy as np
import matplotlib.pyplot as plt


## Plot
def plot_2d(G1: StarlikeCurve, G2: StarlikeCurve, artG1: StarlikeCurve, artG2: StarlikeCurve):
    fig_plot, ax_plot = plt.subplots()
    ax_plot.plot(G1.x, G1.y, 'b--')
    ax_plot.plot(G2.x, G2.y, 'r--')
    ax_plot.plot(artG1.x, artG1.y, 'bo')
    ax_plot.plot(artG2.x, artG2.y, 'ro')

    ax_plot.axis('equal')
    plt.grid()
    plt.show()
    plt.close()


if __name__ == '__main__':
    N=16
    coll_2d_closed = collocation_points_2d(N, startpoint=True)
    point_circle_closed = starlike_circle_base(coll_2d_closed)

    # coll_2d = collocation_points_2d(N, startpoint=False)
    # point_circle = starlike_circle_base(coll_2d)

    g1_r_values = np.ones(coll_2d_closed.n)*2
    Gamma1 = starlike_curve(g1_r_values, point_circle_closed)

    g2_r_values = np.ones(coll_2d_closed.n)*3
    Gamma2 = starlike_curve(g2_r_values, point_circle_closed)


    eta1 = 0.5
    artGamma1 = starlike_curve(g1_r_values*eta1, point_circle_closed)

    eta2 = 2
    artGamma2 = starlike_curve(g2_r_values*eta2, point_circle_closed)

    plot_2d(Gamma1, Gamma2, artGamma1, artGamma2)

    # plot_2d(point_circle)
    # plot_2d(point_circle_closed)

    # eta1 = 0.5
    # art_inner_r_values = np.ones(coll_2d_closed.n)*eta1
    # art_inner_curve = starlike_curve(art_inner_r_values, point_circle_closed)

    # eta2 = 2
    # art_outter_r_values = np.ones(coll_2d_closed.n)*eta2
    # art_outter_curve = starlike_curve(art_outter_r_values, point_circle_closed)


    # r_values = np.ones(N)*2

    # curve = starlike_curve(r_values, point_circle)
    # print(curve)
    # print(curve.shape)
    # print(coll_2d)
    # coll_2d_pickled = cloudpickle.dumps(coll_2d)
    # coll_2d_unpickled = cloudpickle.loads(coll_2d_pickled)
    # print(coll_2d_unpickled)