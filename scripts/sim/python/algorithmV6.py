import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt
from sympy import *
from cvxopt import solvers, matrix
import time
#from torch.autograd.functional import jacobian
from library.visualize_mobile_robot import sim_mobile_robot
def beta(x,y, xc, yc):
    a = 1
    b = 1.1
    function = ((x-xc-a)**2 + (y-yc)**2)*((x-xc+a)**2 + (y-yc)**2) - b**4
    return function

def beta0(x, y, r):
    return x**2 + y**2 - r**2

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
    # J11*J12
    #J = matrix([J11, J12])
    #Jacobian = matrix([[diff(F[i], px) for i in range(0,1)],
    #                   [diff(F[i], py) for i in range(0,1)]])
    J = [J11, J12, J21, J22]
    return J


def main():
    # Constants and Settings
    Ts = .1 # Update simulation every 10ms
    t_max = 120.0 # total simulation duration in seconds
    # Set initial state
    IS_SHOWING_2DVISUALIZATION = True
    MAX_ANGULAR_VEL = 2.84
    MAX_LINEAR_VEL = 0.5
    SIM_RW = True
    SIM_BOTH_WORLDS = True
    OMNI = True

    field_x = (-2.5, 2.5)
    field_y = (-2.5, 2.5)

    px, py, xg1, xg2, r0, r1, r2, x1x, x1y, x2x, x2y, ri0, ri1, ri2 = symbols('px py xg1 xg2 r0 r1 r2 x1x x1y x2x x2y ri0 ri1 ri2')
    q0x, q0y, q1x, q1y, q2x, q2y = symbols('q0x q0y q1x q1y q2x q2y')
    qgx, qgy = symbols('qgx qgy')

    ux = 0.1
    uy = 0.1
    Kp = 1
    l = 0.06
    Kappa = 1
    gamma = 100
    q_t0 = np.array([0., -0.])
    r_t0 = 1.0
    t = 0.0
    qi = q_t0
    q = np.array([-2, -1.0])
    r = r_t0
    r0_t0 = 5.0
    r0value = r0_t0
    q0 = np.array([0., 0.])

    x = q
    theta_robot = 0.0
    theta_robot_bw = theta_robot
    xi = np.array([0., 3.])
    qiF = [q0, qi]
    rho_i = [r0value, r]
    obstacleCount = 1
    M = obstacleCount + 1
    x_g = np.array([2, 2, 0.0])
    q_g = x_g

    Jacobian = calculateJacobian(M)
    J11 = Jacobian[0]   # dF1/dx
    J12 = Jacobian[1]   # dF1/dy
    J21 = Jacobian[2]   # dF2/dx
    J22 = Jacobian[3]   # dF2/dy
    #J = calculateJacobian(Fsym)
    #F = diffeomorphismF(M, x, xi, x_g, rho_i, qiF, q_g)

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        if SIM_RW:
            if OMNI: 
                sim_visualizer = sim_mobile_robot( 'omnidirectional', 1 )
            else: 
                sim_visualizer = sim_mobile_robot( 'unicycle' )
            sim_visualizer.set_field( field_x, field_y ) # set plot area
            sim_visualizer.show_goal(x_g)
            x_values = np.linspace(-5, 5, 100)
            y_values = np.linspace(-5, 5, 100)
            x_mesh, y_mesh = np.meshgrid(x_values, y_values)
            xc1 = q_t0[0]
            yc1 = q_t0[1]
            beta_values = beta(x_mesh, y_mesh, xc1, yc1)
            sim_visualizer.ax.contour(x_mesh, y_mesh, beta_values, levels=[0], colors='r')

        if SIM_BOTH_WORLDS:
            if OMNI:
                sim_visualizer_bw = sim_mobile_robot( 'omnidirectional', 2 )
            else: 
                sim_visualizer_bw = sim_mobile_robot( 'unicycle' )
            sim_visualizer_bw.set_field( field_x, field_y ) # set plot area
            sim_visualizer_bw.show_goal(q_g)
            bw_obst = plt.Circle((q_t0), r_t0, color='r', fill=False)
            bw_safe_set = plt.Circle((q0), r0_t0, color='b', fill=False)
            sim_visualizer_bw.ax.add_patch(bw_obst)
            sim_visualizer_bw.ax.add_patch(bw_safe_set)

    state_history = np.zeros((1000, 2))
    state_history_q = np.zeros((1000, 2))

    step = 0
    while t < t_max:
        state_history[step] = x
        state_history_q[step] = q

        # 4 (INCOMPLETE)

        # Koko jacobian kertaa u

        #J11s = J11.subs({px:1.0, py:2.0, xg1:1.0, xg2:1.0, r0:10.0, r1:2.0, x1x:0.1, x1y:0.3, x2x:2.1, x2y:1.0, ri0:1.1, ri1:1.1, q01:1.1, q11:1.1, qg1:2.1})
        
        # ri0 ja ri1 !!!
        q_dot_elapsed = time.perf_counter()
        J11s = J11.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        J12s = J12.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        J21s = J21.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        J22s = J22.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        print(f'Time elapsed step 4 (jacobian substitution): {time.perf_counter() - q_dot_elapsed}')
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

        Q = matrix([[2.0, 0.0, 0.0, 0.0,],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0*Kappa, 0.0],
                    [0.0, 0.0, 0.0, 2.0*Kappa]])
        
        c = matrix([-2*u_hat[0], -2*u_hat[1], -2*u_hat[2], -2*u_hat[3]])
        
        # u = [uqx uqy ur ur0]
        G = matrix([[AC1[0], AC1[1], AC1[2], 0.0],
                    [0.0, 0.0, 0.0, AC0],
                    [AC3[0], AC3[1], AC3[2], AC3[3]]]).T

        h = matrix([bC1, bC0, bC3])
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, c, G, h, verbose=False)

        u_star_qx = sol['x'][0]
        u_star_qy = sol['x'][1]
        u_star_r = sol['x'][2]
        u_star_r0 = sol['x'][3]


        # 7
        
        qi[0] += u_star_qx*Ts
        qi[1] += u_star_qy*Ts
        r += u_star_r*Ts
        r0value += u_star_r0*Ts
        rho_i = [r0value, r]

        t += Ts

        # 8
        jacobian_substitution_elapsed = time.perf_counter()

        J11s = J11.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        J12s = J12.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        J21s = J21.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        J22s = J22.subs({px:x[0], py:x[1], xg1:x_g[0], xg2:x_g[1], r0:rho_i[0], r1:rho_i[1], x1x:xi[0], x1y:xi[1], q0x:q0[0], q0y:q0[1], q1x:qi[0], q1y:qi[1], qgx:q_g[0], qgy:q_g[1]})
        print(f'Time elapsed step 8 (Jacobian substitution): {time.perf_counter() - jacobian_substitution_elapsed }')

        jacobian = np.array([[J11s, J12s], [J21s, J22s]], dtype=float)
        inv_jacobian = np.linalg.inv( jacobian )
        x_dot = inv_jacobian @ q_dot

        ux = x_dot[0]
        uy = x_dot[1]
        if OMNI:
            control_input = np.array([x_dot[0], x_dot[1]])
            control_input_bw = np.array([q_dot[0], q_dot[1]])
            control_input[0] = min(max(control_input[0], -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
            control_input[1] = min(max(control_input[1], -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
            control_input_bw[0] = min(max(control_input_bw[0], -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
            control_input_bw[1] = min(max(control_input_bw[1], -MAX_LINEAR_VEL), MAX_LINEAR_VEL)


        else:
            vx_bw = q_dot[0] * np.cos(theta_robot_bw) + q_dot[1] * np.sin(theta_robot_bw)
            wz_bw = (- q_dot[0] * np.sin(theta_robot_bw) + q_dot[1] * np.cos(theta_robot_bw)) / l

            vx_bw = min(max(vx_bw, -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
            wz_bw = min(max(wz_bw, -MAX_ANGULAR_VEL), MAX_ANGULAR_VEL)

            vx = ux * np.cos(theta_robot) + uy * np.sin(theta_robot)
            wz = (- ux * np.sin(theta_robot) + uy * np.cos(theta_robot)) / l

            # Confine cmd values to upper and lower bounds
            vx = min(max(vx, -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
            wz = min(max(wz, -MAX_ANGULAR_VEL), MAX_ANGULAR_VEL)

            control_input = np.array([vx, wz])
            control_input_bw = np.array([vx_bw, wz_bw])
        
        # 9
        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            if SIM_RW:
                sim_visualizer.update_time_stamp(t)
                sim_visualizer.update_goal( x_g )
                sim_visualizer.update_trajectory( state_history[:step+1] ) # up to the latest data
            if SIM_BOTH_WORLDS:
                sim_visualizer_bw.update_time_stamp(t)
                sim_visualizer_bw.update_goal( q_g )
                sim_visualizer_bw.update_trajectory( state_history_q[:step+1] ) # up to the latest data
                bw_obst.set_center(qi)
                bw_obst.set_radius(r)
                bw_safe_set.set_radius(r0value)
            plt.pause(0.000001)  

        # Update bw
        if OMNI:
            x = x + Ts*control_input
            q = q + Ts*control_input_bw
            x = x.astype(np.float64)
            q = q.astype(np.float64)
    
        else:
            B = np.array([[np.cos(theta_robot_bw), 0], [np.sin(theta_robot_bw), 0], [0, 1]])
            q_step = Ts*(B @ control_input_bw)
            q = q + q_step[:1] # will be used in the next iteration
            theta_robot_bw = float(q_step[2])
            theta_robot_bw = ( (theta_robot_bw  + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]
        # Update rw
 
            B = np.array([[np.cos(theta_robot), 0], [np.sin(theta_robot), 0], [0, 1]])
            x_step = Ts*(B @ control_input) # will be used in the next iteration
            x = x + x_step[:1]
            theta_robot = float(x_step[2])
            theta_robot = ( (theta_robot + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        step += 1

main()
