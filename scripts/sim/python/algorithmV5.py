import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt
from sympy import *
from cvxopt import solvers, matrix
#from torch.autograd.functional import jacobian
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
    F_list = []

    beta_i = []
    beta_i.append((x[0]-xi[0][0])**2 + (x[1]-xi[0][1])**2 - rho_i[0]**2)
    for i in range(1,M):
        beta_i.append(((x[0]-xi[i][0]-a)**2 + (x[1]-xi[i][1])**2)*((x[0]-xi[i][0]+a)**2 + (x[1]-xi[i][1])**2) - b**4)
    
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
        F_list.append(l)
    sigma_g = 1 - sigma
    F_list.append(sigma_g * (x-x_g+q_g))
    F = sum(F_list)
    #print(sigma_g * (x-x_g+q_g))
    #print(F)
    return F


def calculateJacobian(M):
    px, py, xg1, xg2, r0, r1, r2, x1x, x1y, x2x, x2y, ri0, ri1, ri2 = symbols('px py xg1 xg2 r0 r1 r2 x1x x1y x2x x2y ri0 ri1 ri2')
    q0x, q0y, q1x, q1y, q2x, q2y = symbols('q0x q0y q1x q1y q2x q2y')
    qgx, qgy = symbols('qgx qgy')
    r = symbols('r0:{}'.format(M))
    r = Matrix(r)
    #for i in range(0,M):
    #    print(r[i])
    rvalues = [10.0, 2.0, 2.5]
    rs = [r[i].subs({r[i]: rvalues[i]}) for i in range(M)]
    #print(rs)
    lam = 100
    a = 1
    b = 1.1
    x = np.array([px, py])
    xg = np.array([xg1, xg2])
    ri = [ri0, ri1, ri2]
    qi = [np.array([q0x, q0y]), np.array([q1x, q1y]), np.array([q2x, q2y])]
    qg = np.array([qgx, qgy])
    if M == 2:
        xi = [np.array([0., 0.]), np.array([x1x, x1y])]
        rho = [r0, r1, r2]
    if M == 3:
        xi = [np.array([0., 0.]), np.array([x1x, x1y]), np.array([x2x, x2y])]
    theta_i = []
    for i in range(0,M):
        theta_i.append(atan2(x[0]-xi[i][0],x[1]-xi[i][1]))
    beta_i = []
    beta0 = (x[0]-xi[0][0])**2 + (x[1]-xi[0][1])**2 - r0**2
    beta_i.append(beta0)
    for i in range(1,M):
        beta_i.append(((x[0]-xi[i][0]-a)**2 + (x[1]-xi[i][1])**2)*((x[0]-xi[i][0]+a)**2 + (x[1]-xi[i][1])**2) - b**4)
    beta_dash_i = {}
    for i in range(0,M):
        p = []
        for j in range(0,M):
            if i != j:
                p.append(beta_i[j])
        beta_dash_i[i] = np.prod(p)
    gamma = (sqrt((x[0]-xg[0])**2 + (x[1]-xg[1])**2))**2
    F_list = []
    sigma = []
    #(np.linalg.norm(x-xi)/(1+beta_i[i]))
    for i in range(0,M):
        sigma_i = (gamma*beta_dash_i[i])/(gamma*beta_dash_i[i]+lam*beta_i[i])
        sigma.append(sigma_i)
        #f = ((sqrt((x[0]-xi[i][0])**2 + (x[1]-xi[i][1])**2))/ri[i])*np.array([cos(theta_i[i]), sin(theta_i[i])]).T
        f = ( (sqrt((x[0]-xi[i][0])**2 + (x[1]-xi[i][1])**2)) / ((sqrt((x[0]-xi[i][0])**2 + (x[1]-xi[i][1])**2))/(1+beta_i[i])) ) * np.array([cos(theta_i[i]), sin(theta_i[i])]).T 
        l = sigma_i*(rho[i]*f+qi[i])
        F_list.append(l)
    sigmag = 1 - sum(sigma)
    F_list.append(sigmag*(x-xg-qg))
    F = sum(F_list)
    J11 = diff(F[0],px)
    #print(J11)
    # px py xg1 xg2 r0 r1 r2 xi1 xi2 xi11 xi22 ri0 ri1 ri2
    # q01 q02 q11 q12 q21 q22
    # qg1 qg2
    #J11s = J11.subs({px:1.0, py:2.0, xg1:1.0, xg2:1.0, r0:10.0, r1:2.0, x1x:0.1, x1y:0.3, x2x:2.1, x2y:1.0, ri0:1.1, ri1:1.1, q01:1.1, q11:1.1, qg1:2.1})
    #print(J11s)
    J12 = diff(F[0],py)
    J21 = diff(F[1],px)
    J22 = diff(F[1],py)
    J11*J12
    #J = matrix([J11, J12])
    #Jacobian = matrix([[diff(F[i], px) for i in range(0,1)],
    #                   [diff(F[i], py) for i in range(0,1)]])
    J = [J11, J12, J21, J22]
    return J


def main():
    # Constants and Settings
    Ts = 0.01 # Update simulation every 10ms
    t_max = 12.0 # total simulation duration in seconds
    # Set initial state

    px, py, xg1, xg2, r0, r1, r2, x1x, x1y, x2x, x2y, ri0, ri1, ri2 = symbols('px py xg1 xg2 r0 r1 r2 x1x x1y x2x x2y ri0 ri1 ri2')
    q0x, q0y, q1x, q1y, q2x, q2y = symbols('q0x q0y q1x q1y q2x q2y')
    qgx, qgy = symbols('qgx qgy')

    ux = 0.1
    uy = 0.1
    Kp = 1
    Kappa = 1
    gamma = 1
    q_t0 = np.array([0., -3.])
    r_t0 = 1.0
    t = 0.0
    qi = q_t0
    q = np.array([1.0, 1.0])
    r = r_t0
    r0_t0 = 5.0
    r0value = r0_t0
    q0 = np.array([0., 0.])

    x = q
    xi = np.array([0., 3.])
    qiF = [q0, qi]
    rho_i = [r0value, r]
    obstacleCount = 1
    M = obstacleCount + 1
    x_g = np.array([3.0, 3.0])
    q_g = x_g

    Jacobian = calculateJacobian(M)
    J11 = Jacobian[0]   # dF1/dx
    J12 = Jacobian[1]   # dF1/dy
    J21 = Jacobian[2]   # dF2/dx
    J22 = Jacobian[3]   # dF2/dy
    #J = calculateJacobian(Fsym)
    #F = diffeomorphismF(M, x, xi, x_g, rho_i, qiF, q_g)

    while t < t_max:

        # 4 (INCOMPLETE)

        # Koko jacobian kertaa u

        #J11s = J11.subs({px:1.0, py:2.0, xg1:1.0, xg2:1.0, r0:10.0, r1:2.0, x1x:0.1, x1y:0.3, x2x:2.1, x2y:1.0, ri0:1.1, ri1:1.1, q01:1.1, q11:1.1, qg1:2.1})
        
        # ri0 ja ri1 !!!
        J11s = J11.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        J12s = J12.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        J21s = J21.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        J22s = J22.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        q_dot = np.array([J11s*ux+J12s*uy, J21s*ux+J22s*uy]).T

        # 5

        u_hat_q = Kp*(q_t0-qi)
        u_hat_r = Kp*(r_t0-r)
        u_hat_r0 = Kp*(r0_t0-r0value)
        u_hat = matrix([u_hat_q[0], u_hat_q[1], u_hat_r, u_hat_r0])

        # 6

        # C1 + C0
        hi = (np.linalg.norm(qi-q))**2 - r**2
        h0 = r0value**2 - (np.linalg.norm(q0-q))**2
        AC1 = matrix([[-2*(qi[0]-q[0]), -2*(qi[1]-q[1]), 2*r]]).T
        #print(AC1)
        AC0 = -2*r0value
        bC1 = float(-2*(qi-q) @ q_dot + gamma * hi)
        bC0 = float(2*(q0-q) @ q_dot + gamma * h0)

        # C2
        #hij = np.linalg.norm(qi-qj)**2 - (ri-rj)**2
        #AC2 = matrix([[-2*(qi[0]-qj[0]), -2*(qi[1]-qj[1]), 2*(qi[0]-qj[0]), 2*(qi[1]-qj[1]), 2*(ri+rj), 2*(ri+rj)]]).T
        #bC2 = gamma*hij

        # C3
        hi0 = (r0value-r)**2 - np.linalg.norm(qi-q0)**2
        AC3 = matrix([[2*(qi[0]-q0[0]), 2*(qi[1]-q0[1]), 2*(r0value-r), -2*(r0value-r)]])
        bC3 = float(gamma*hi0)

        a = matrix([2.0, 2.0, 2*Kappa, 2*Kappa])
        A = np.identity(4)
        Q = a*A
        c = matrix([-2*u_hat[0], -2*u_hat[1], -2*u_hat[2], -2*u_hat[3]])
        
        # u = [uqx uqy ur ur0]
        G = matrix([[AC1[0], AC1[1], AC1[2], 0.0],
                    [0.0, 0.0, 0.0, AC0],
                    [AC3[0], AC3[1], AC3[2], AC3[3]]])

        print(type(AC1[0]))
        h = matrix([bC1, bC0, bC3]).T
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, c, G, h, verbose=False)
        #uq1 = sol['x'][0]
        #uq2 = sol['x'][1]
        u_star_qx = 0.1
        u_star_qy = 0.1
        u_star_r = 0.1
        u_star_r0 = 0.1

        # 7
        
        qi[0] += u_star_qx*Ts
        qi[1] += u_star_qy*Ts
        r += u_star_r*Ts
        r0value += u_star_r0*Ts
        rho_i = [r0value, r]

        t += Ts
        #print(u_hat_q)
        #print(u_hat_r)

        # 8

        # ...

        # 9

        # ...
        print("1 round done")

main()
