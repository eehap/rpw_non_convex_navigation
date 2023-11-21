import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
from library.visualize_mobile_robot import sim_mobile_robot

def diffeomorphismF(x, xi, x_g):
    ri = 1
    fi = np.linalg.norm(x-xi)/ri
    F = 1
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
xi = q_t0
M = 1
x_g = np.array([3.0, 3.0])


while t < t_max:

    # INCOMPLETE
    F = diffeomorphismF(x, xi, x_g)

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
