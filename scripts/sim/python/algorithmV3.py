import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin
from cvxopt import solvers, matrix
from library.visualize_mobile_robot import sim_mobile_robot


def calculate_r():
    r = 1.0
    return r


def calculate_theta(x, xi):
    theta = atan2(x[0]-xi[0], x[1]-xi[1])
    return theta


def diffeomorphismF(M, x, xi, x_g, rho_i, qi, q_g):
    lam = 100
    a = 1
    b = 1.1
    gamma = (np.linalg.norm(x-x_g))**2
    F = np.array([0., 0.])

    beta_i = []
    for i in range(0,M):
        beta_i.append(((x[0]-xi[i][0]-a)**2 + (x[1]-xi[i][1])**2)*((x[0]-xi[i][0]+a)**2 + (x[1]-xi[i][0])**2) - b**4)

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
        ri = calculate_r()
        fi = (np.linalg.norm(x-xi)/ri)*np.array([cos(theta), sin(theta)])
        l = sigma_i*(rho_i[i]*fi+qi[i])
        #print(sigma_i, " ", rho_i[i], " ", fi, " ", qi[i])
        #print(l)
        F += l
        #print(F)
    sigma_g = 1 - sigma
    F += sigma_g * (x-x_g+q_g)
    #print(F)
    return F


# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 12.0 # total simulation duration in seconds
# Set initial state

u_hat_q = np.array([0., 0.])
u_hat_r = 0.0
Kp = 1
Kappa = 1
gamma = 1
q_t0 = np.array([2.0, 1.5])
r_t0 = 1.0
t = 0.0
qi = q_t0
q = np.array([1.0, 1.0])
r = r_t0
q_dot = np.array([0.3, 0.1])
r0 = 5.0
q0 = np.array([0., 0.])

x = q
xi = [q0, qi]
qiF = [q0, qi]
rho_i = [r0, r]
obstacleCount = 1
M = obstacleCount + 1
x_g = np.array([3.0, 3.0])
q_g = x_g

F = diffeomorphismF(M, x, xi, x_g, rho_i, qiF, q_g)

while t < t_max:

    # 4 (INCOMPLETE)
    #F = diffeomorphismF(obstacleCount, x, xi, x_g)

    # 5
    u_hat_q = Kp*(q_t0-qi)
    u_hat_r = Kp*(r_t0-r)

    # 6

    # C1 + C0
    hi = (np.linalg.norm(qi-q))**2 - r**2
    h0 = r0**2 - (np.linalg.norm(q0-q))**2
    AC1 = matrix([[-2*(qi[0]-q[0]), -2*(qi[1]-q[1]), 2*r]]).T
    AC0 = -2*r0
    bC1 = -2*(qi-q).T @ q_dot + gamma * hi
    bC0 = 2*(q0-q).T @ q_dot + gamma * h0

    # C2
    #hij = np.linalg.norm(qi-qj)**2 - (ri-rj)**2
    #AC2 = matrix([[-2*(qi[0]-qj[0]), -2*(qi[1]-qj[1]), 2*(qi[0]-qj[0]), 2*(qi[1]-qj[1]), 2*(ri+rj), 2*(ri+rj)]]).T
    #bC2 = gamma*hij

    # C3
    hi0 = (r0-r)**2 - np.linalg.norm(qi-q0)**2
    AC3 = matrix([[2*(qi[0]-q0[0]), 2*(qi[1]-q0[1]), 2*(r0-r), -2*(r0-r)]])
    bC3 = gamma*hi0

    # TODO
    # How to add all of these together (G+h)? (different x)
    # Need to add compatibility with multiple objects...
    # Using the solvers output for u_star_q and u_star_r

    u_star_q = np.array([0.1, -0.2])
    u_star_r = 0.1

    Q = matrix([[2, 0, 0], [0, 2, 0], [0, 0, 2*Kappa]])
    c = matrix([-2*u_hat_q[0], -2*u_hat_q[1], -2*Kappa*u_hat_r])
    #G = matrix([[2*(qi[0]-q[0]), 2*(qi[1]-q[1]), 0]])
    #print(AC1)
    #print()
    #sol = solvers.qp(Q, c, G, h, verbose=False)
    #uq1 = sol['x'][0]
    #uq2 = sol['x'][1]
    #u_star_q = [uq1, uq2] ??

    # 7
    qi += u_star_q*Ts
    r += u_star_r*Ts

    t += Ts
    #print(u_hat_q)
    #print(u_hat_r)
