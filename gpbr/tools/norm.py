import numpy as np
from scipy import integrate

def l2_norm_relative_trapz(exact, approx):
    return np.sqrt(integrate.trapezoid((exact-approx)**2, axis=0)/integrate.trapezoid(exact**2, axis=0))


# def calc_l2_norm(theta_coll, phi_coll, t_coll, exact, approx, approxfunc=None):
#     thetas = np.linspace(0, np.pi, 512)
#     phis = np.linspace(0, 2*np.pi, 512)
#     Theta, Phi = np.meshgrid(thetas, phis)
#     approx_vals = approxfunc(Theta, Phi)
#     if np.ndim(approx_vals) == 0:
#         approx_vals = np.full(Theta.shape, approx_vals)
#     # return np.linalg.norm((exact - approx)**2)
#     return np.linalg.norm((exact - approx))**2 + + integrate.trapezoid(integrate.trapezoid(approx_vals**2, thetas, axis=0), phis, axis=0)*1e-4