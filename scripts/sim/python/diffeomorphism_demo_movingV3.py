import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from math import sin, cos, atan2, sqrt


def calculate_theta(x, xi):
    theta = atan2(x[1]-xi[1], x[0]-xi[0])
    return theta


def diffeomorphismF(M, x, xi, x_g, rho_i, qi, q_g):
    ri = [5.0, 0.5]
    lam = 100
    #a = 1
    #b = 1.1
    gamma = (np.linalg.norm(x-x_g))**2
    F_list = []

    beta_i = []
    # Beta0
    beta_i.append(rho_i[0]**2 - np.linalg.norm(x-xi[0])**2)
    # Beta1
    beta_i.append(np.linalg.norm(x-xi[1])**2 - rho_i[1]**2)
    # Beta2
    #beta_i.append((x[0]-xi[2][0])**2 + (x[1]-xi[2][1])**2 - rho_i[2]**2)

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


def numerical_jacobian(q_robot, obst, radii, goal, lam):
    gradient_beta_0 = np.array([-2*obst[0][0], -2*obst[0][1]])
    gradient_beta_1 = np.array([-2*obst[1][0], -2*obst[1][1]])

    beta_0 = radii[0] ** 2 - norm(q_robot - obst[0]) ** 2
    beta_1 = norm(q_robot - obst[1]) ** 2 - radii[1] ** 2

    v0 = radii[0] * (1 - beta_0) / norm(q_robot - obst[0])
    v1 = radii[1] * (1 + beta_1) / norm(q_robot - obst[1])

    gradient_v0 =  v0 * (norm(q_robot - obst[0]) / (1 + beta_0) * gradient_beta_0 - (1 / norm(q_robot - obst[0])) * q_robot - obst[0])
    gradient_v1 =  v1 * (norm(q_robot - obst[1]) / (1 + beta_1) * gradient_beta_1 - (1 / norm(q_robot - obst[1])) * q_robot - obst[1])

    gamma_d = norm(q_robot - goal[:1]) ** 2
    # -2*py*((px - xg1)**2 + (py - xg2)**2) + (2*py - 2*xg2)*(-px**2 - py**2 + r0**2)
    gradient_gamma_d_beta_dash_1 = np.array([-2*q_robot[0]*((q_robot[0] - goal[0]) ** 2) + ((q_robot[1] - goal[1])**2) + (2*q_robot[0] - 2*goal[0])*(-q_robot[0]**2 - q_robot[1]**2 + radii[0]**2),
                                             -2*q_robot[1]*((q_robot[0] - goal[0]) ** 2) + ((q_robot[1] - goal[1])**2) + (2*q_robot[1] - 2*goal[1])*(-q_robot[0]**2 - q_robot[1]**2 + radii[0]**2)])
    # (2*py - 2*x1y)*((px - xg1)**2 + (py - xg2)**2) + (2*py - 2*xg2)*(-r1**2 + (px - x1x)**2 + (py - x1y)**2)
    gradient_gamma_d_beta_dash_0 = np.array([(2*q_robot[0] - 2*obst[1][0])*((q_robot[0] - goal[0])**2 + (q_robot[1] - goal[1])**2) + (2*q_robot[0] - 2*goal[0])*(-radii[1]**2 + (q_robot[0] - obst[1][0])**2 + (q_robot[1] - obst[1][1])**2),
                                             (2*q_robot[1] - 2*obst[1][1])*((q_robot[0] - goal[0])**2 + (q_robot[1] - goal[1])**2) + (2*q_robot[1] - 2*goal[1])*(-radii[1]**2 + (q_robot[0] - obst[1][0])**2 + (q_robot[1] - obst[1][1])**2)])
    beta_dash_0 = beta_1
    beta_dash_1 = beta_0

    sigma_0 = gamma_d * beta_dash_0 / (gamma_d * beta_dash_0 + lam * beta_0)
    sigma_1 = gamma_d * beta_dash_1 / (gamma_d * beta_dash_1 + lam * beta_1)

    sigma_d = 1 - (sigma_0 + sigma_1)

    gradient_sigma_0 = (lam/((gamma_d*beta_dash_0+lam*beta_0)**2))*(beta_0 * gradient_gamma_d_beta_dash_0 - gamma_d*beta_dash_0*gradient_beta_0)
    gradient_sigma_1 = (lam/((gamma_d*beta_dash_1+lam*beta_1)**2))*(beta_1 * gradient_gamma_d_beta_dash_1 - gamma_d*beta_dash_1*gradient_beta_1)
    
    
    J = (sigma_0 * v0 * np.eye(2) + sigma_0 * (q_robot - obst[0]) * gradient_v0.T + (v0 - 1) * (q_robot - obst[0]) * gradient_sigma_0.T +
         sigma_1 * v1 * np.eye(2) + sigma_1 * (q_robot - obst[1]) * gradient_v1.T + (v1 - 1) * (q_robot - obst[1]) * gradient_sigma_1.T) + sigma_d * np.eye(2)

    return J


grid_size = 7
x_initial, y_initial = np.meshgrid(np.linspace(-2.5, 2.5, grid_size), np.linspace(-2.5, 2.5, grid_size))
x_initial = x_initial.flatten()
y_initial = y_initial.flatten()
xi = np.array([[0., 0.], [1., 1.], [-1., -1.]])
rho_i = np.array([5.0, 0.5, 0.5])
qi = np.array([[0., 0.], [1., 1.], [-1., -1.]])

def update_positions(x_initial, y_initial, xi, rho_i, qi, useInverse):
    x_ = np.array([-2.5, -2.5])
    print(x_)
    #qi = xi
    x_g = np.array([2., 2.])
    #M = len(xi)
    M = 2
    q_g = x_g
    x_updated = np.zeros((len(x_initial)))
    y_updated = np.zeros((len(y_initial)))
    for i in range(len(x_initial)):
        x_ = np.array([x_initial[i], y_initial[i]])
        if useInverse:
            # q_robot, obst, radii, goal, lam
            lam = 100
            for i in range(0, len(x_initial)-1):
                x = np.array([[0.0, 0.0], [x_initial[i], y_initial[i]]])
                j = numerical_jacobian(x_, x, rho_i, x_g, lam)
                inv_jacobian = np.linalg.inv(j)
                x = np.array([x_initial[i], y_initial[i]])
                a = inv_jacobian @ x
                x_updated[i] = a[0]
                y_updated[i] = a[1]
        else:
            F = diffeomorphismF(M, x_, xi, x_g, rho_i, qi, q_g)
            x_updated[i] = F[0]
            y_updated[i] = F[1]
    return x_updated, y_updated

x_updated, y_updated = update_positions(x_initial, y_initial, xi, rho_i, qi, False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

ax1.scatter(x_initial, y_initial, color='blue', label='Initial Positions')
ax1.add_patch(plt.Circle((xi[1][0], xi[1][1]), rho_i[1], color='red', label='obstacle'))
ax1.set_title('Initial Positions')
ax1.set_xlim([-4, 4])
ax1.set_ylim([-4, 4])

scatter = ax2.scatter([], [], color='red', label='Updated Positions')
obstacle = plt.Circle((xi[1][0], xi[1][1]), rho_i[1], color='red', label='obstacle')
#obstacle2 = plt.Circle((xi[2][0], xi[2][1]), rho_i[2], color='green', label='obstacle')
ax2.add_patch(obstacle)
#ax2.add_patch(obstacle2)
ax2.set_title('Updated Positions')
ax2.set_xlim([-4, 4])
ax2.set_ylim([-4, 4])

scatter2 = ax3.scatter(x_initial, y_initial, color='blue', label='Diff + Inverse')
#ax3.add_patch(plt.Circle((xi[1][0], xi[1][1]), rho_i[1], color='red', label='obstacle'))
ax3.set_title('Diff + Inverse')
ax3.set_xlim([-4, 4])
ax3.set_ylim([-4, 4])

arrows = []
for i in range(grid_size**2):
    arrow = ax2.arrow(x_initial[i], y_initial[i], x_updated[i] - x_initial[i], y_updated[i] - y_initial[i],
                      color='black', shape='full', lw=0.1, length_includes_head=True, head_width=0.1)
    arrows.append(arrow)

def update_plot(xi, scatter, scatter2, obstacle, arrows):
    qi[1][0] += np.random.randn() * 0.1
    qi[1][1] += -0.05 + np.random.randn() * 0.01
    rho_i[1] += np.random.randn() * 0.1
    obstacle.set_radius(rho_i[1])
    x_updated, y_updated = update_positions(x_initial, y_initial, xi, rho_i, qi, False)
    x_updated2, y_updated2 = update_positions(x_updated, y_updated, xi, rho_i, qi, True)
    obstacle.set_center((qi[1][0], qi[1][1]))
    scatter.set_offsets(np.column_stack((x_updated, y_updated)))
    scatter2.set_offsets(np.column_stack((x_updated2, y_updated2)))
    obstacle.set_center((qi[1][0], qi[1][1]))

    for i in range(grid_size**2):
        new_arrow = ax2.arrow(x_initial[i], y_initial[i], x_updated[i] - x_initial[i], y_updated[i] - y_initial[i],
                          color='black', shape='full', lw=0.1, length_includes_head=True, head_width=0.1)
        arrows[i].remove()
        arrows[i] = new_arrow
    plt.pause(0.1)

i = 0
while i < 100:
    update_plot(xi, scatter, scatter2, obstacle, arrows)
    plt.pause(0.1)
    ax2.figure.canvas.draw()
    i += 1

#plt.tight_layout()
#plt.show()
