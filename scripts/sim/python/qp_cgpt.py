import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 12.0 # total simulation duration in seconds
# Set initial state

Kp = 1.0
q_t0 = np.array([2.0, 1.5])
qi = np.array([1.0, 2.0])
kappa = 1.0
gamma = 1.0
r_t0 = 1.0
r = r_t0
qdot = np.array([1.0, 0.4])
q0 = np.array([0.0, 0.0])
r0 = 5.0

u_hat_q = Kp*(q_t0-qi)
u_hat_r = Kp*(r_t0-r)

P = matrix([[2, 0, 0], [0, 2, 0], [0, 0, 2*kappa]])
q = matrix([-2*u_hat_q[0], -2*u_hat_q[1], -2*kappa*u_hat_r])

G = matrix([[2*(qi[0]-q[0]), 2*(qi[1]-q[1]), 2*u_hat_r]]).T
h = gamma*matrix([r**2])

G0 = matrix([[2*(q0[0]-q[0]), 2*(q0[1]-q[1]), 2*r0]]).T
h0 = matrix([r0**2])

Ac1 = matrix([[-2*(qi[0]-q[0]), -2*(qi[1]-q[1]), 2*u_hat_r]]).T
bc1 = matrix([-2*(qi[0]-q[0]) * qdot[0] - 2*(qi[1]-q[1]) * qdot[1] + h[0]])

Ac0 = matrix([[2*(q0[0]-q[0]), 2*(q0[1]-q[1]), 2*r0]]).T
bc0 = matrix([2*(q0[0]-q[0]) * qdot[0] + 2*(q0[1]-q[1]) * qdot[1] + h0[0]])

sol = solvers.qp(P, q, G=G, h=h, A=Ac1, b=bc1, G0=G0, h0=h0, A0=Ac0, b0=bc0)

u_star_q = sol['x'][0]
u_star_r = sol['x'][1]
print(u_star_q, " ", u_star_r)