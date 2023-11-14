import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
from library.visualize_mobile_robot import sim_mobile_robot

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

while t < t_max:

    # 5
    u_hat_q = Kp*(q_t0-qi)
    u_hat_r = Kp*(r_t0-r)

    # 6
    h_o = (np.linalg.norm(qi-q))**2 - r**2
    h = gamma*h_o
    #AC1 = np.array([-2*(qi-q).T, 2*r])
    AC1 = matrix([[-2*(qi[0]-q[0]), -2*(qi[1]-q[1]), 2*r]])
    #bC1 = -2*(qi-q).T*q_dot
    bC1 = matrix([[-2*(qi[0]-q[0])*q_dot[0]-2*(qi[1]-q[1]*q_dot[1]+h)]])

    u_star_q = np.array([0.1, -0.2])
    u_star_r = np.array([0.3, 0.1])

    Q = matrix([[2, 0], [0, 2*Kappa]])
    #c = matrix([-2*u_hat_q[0], -2*u_hat_q[1], -2*Kappa*u_hat_r])
    G = matrix([[2*(qi[0]-q[0]), 2*(qi[1]-q[1]), 0]])
    #print(AC1)
    #print()
    #sol = solvers.qp(Q, c, G, h, verbose=False)
    #ux = sol['x'][0]
    #uy = sol['x'][1]

    # 7
    qi += u_star_q*Ts
    r += u_star_r*Ts

    t += Ts
    #print(u_hat_q)
    #print(u_hat_r)
