import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from math import sin, cos, atan2, sqrt


def calculate_theta(x, xi):
    theta = atan2(x[1]-xi[1], x[0]-xi[0])
    return theta


def diffeomorphismOld(x_robot, x_obs, x_d, q_obs, q_d, r_obs, lam):
    beta_0 = r_obs[0] ** 2 - norm(x_robot - x_obs[0]) ** 2
    #beta_0 = r_obs[0]**2 - x_robot**2 - x_obs[0]**2
    beta_1 = norm(x_robot - x_obs[1]) ** 2 - r_obs[1] ** 2
    #beta_1 = x_robot**2 - x_obs[1]**2 - r_obs[1]**2

    v0 = r_obs[0] * (1 - beta_0) / norm(x_robot - x_obs[0])
    v1 = r_obs[1] * (1 + beta_1) / norm(x_robot - x_obs[1])

    gamma_d = norm(x_robot - x_d) ** 2

    beta_dash_0 = beta_1
    beta_dash_1 = beta_0

    sigma_0 = gamma_d * beta_dash_0 / (gamma_d * beta_dash_0 + lam * beta_0)
    sigma_1 = gamma_d * beta_dash_1 / (gamma_d * beta_dash_1 + lam * beta_1)

    sigma_d = 1 - (sigma_0 + sigma_1)

    h_lam = sigma_0 * (v0 * (x_robot - x_obs[0]) + q_obs[0]) + sigma_1 * (v1 * (x_robot - x_obs[1]) + q_obs[1]) + sigma_d * (x_robot - x_d + q_d)

    return h_lam


def diffeomorphismF(M, x, xi, x_g, rho_i, qi, q_g):
    ri = [5.0, 0.5]
    lam = 100
    #a = 1
    #b = 1.1
    gamma = (norm(x-x_g))**2
    F_list = []

    beta_i = []
    # Beta0
    beta_i.append(rho_i[0]**2 - norm(x-xi[0])**2)
    #beta_i.append(rho_i[0]**2 - x**2 - xi[0]**2)
    # Beta1
    beta_i.append(norm(x-xi[1])**2 - rho_i[1]**2)
    #beta_i.append(x**2 - xi[1]**2 - rho_i[1]**2)

    beta_dash_i = {}
    if len(beta_i) > 1:
        for i in range(0,M):
            p = []
            for j in range(0,M):
                if i != j:
                    p.append(beta_i[j])
            beta_dash_i[i] = np.prod(p)
    else:
        beta_dash_i[0] = 0.0
    sigma = 0.0
    for i in range(0,M):
        sigma_i = (gamma*beta_dash_i[i])/(gamma*beta_dash_i[i]+lam*beta_i[i])
        sigma += sigma_i
        theta = calculate_theta(x,xi[i])
        fi = (np.linalg.norm(x-xi[i])/ri[i])*np.array([cos(theta), sin(theta)])
        l = sigma_i*(rho_i[i]*fi+qi[i])
        F_list.append(l)
    sigma_g = 1 - sigma
    F_list.append(sigma_g * (x-x_g+q_g))
    F = sum(F_list)
    return F


grid_size = 7
x_initial, y_initial = np.meshgrid(np.linspace(-2.5, 2.5, grid_size), np.linspace(-2.5, 2.5, grid_size))
x_initial = x_initial.flatten()
y_initial = y_initial.flatten()
xi = np.array([[0., 0.], [1., 1.], [-1., -1.]])
rho_i = np.array([5.0, 0.5, 0.5])
qi = np.array([[0., 0.], [1., 1.], [-1., -1.]])
x_g = np.array([2., 2.])
q_g = x_g

def update_positions(x_initial, y_initial, xi, rho_i, qi, x_g, q_g):
    M = 2
    x_updated = np.zeros((len(x_initial)))
    y_updated = np.zeros((len(y_initial)))
    x_updated_old = np.zeros((len(y_initial)))
    y_updated_old = np.zeros((len(y_initial)))
    

    for i in range(len(x_initial)):
        x_ = np.array([x_initial[i], y_initial[i]])
        x_ = x_g
        F = diffeomorphismF(M, x_, xi, x_g, rho_i, qi, q_g)
        x_updated[i] = F[0]
        y_updated[i] = F[1]
        h = diffeomorphismOld(x_, xi, x_g, qi, q_g, rho_i, lam=100)
        x_updated_old[i] = h[0]
        y_updated_old[i] = h[1]
    return x_updated, y_updated, x_updated_old , y_updated_old

x_updated, y_updated, x_updated_old, y_updated_old = update_positions(x_initial, y_initial, xi, rho_i, qi, x_g, q_g)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

ax1.scatter(x_initial, y_initial, color='blue', label='Initial Positions')
ax1.add_patch(plt.Circle((xi[1][0], xi[1][1]), rho_i[1], color='red', label='obstacle'))
ax1.set_title('Initial Positions')
ax1.set_xlim([-5.5, 5.5])
ax1.set_ylim([-5.5, 5.5])

scatter = ax2.scatter(x_updated, y_updated, color='red', label='Updated Positions (original paper)')
obstacle = plt.Circle((xi[1][0], xi[1][1]), rho_i[1], color='red', label='obstacle')
#obstacle2 = plt.Circle((xi[2][0], xi[2][1]), rho_i[2], color='green', label='obstacle')
ax2.add_patch(obstacle)
#ax2.add_patch(obstacle2)
ax2.set_title('Updated Positions (original paper)')
ax2.set_xlim([-5.5, 5.5])
ax2.set_ylim([-5.5, 5.5])

scatter_old_paper = ax3.scatter(x_updated_old, y_updated_old, color='red', label='Updated Positions (old paper)')
obstacle2 = plt.Circle((xi[1][0], xi[1][1]), rho_i[1], color='red', label='obstacle')
ax3.add_patch(obstacle2)
ax3.set_title('Updated Positions (old paper)')
ax3.set_xlim([-5.5, 5.5])
ax3.set_ylim([-5.5, 5.5])

plt.tight_layout()
plt.show()
