import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt
from cvxopt import solvers, matrix
import time 
from library.visualize_mobile_robot import sim_mobile_robot

def numerical_jacobian(x_robot, x_obst, q_radii, goal, lam): 
    beta_0 = q_radii[0] ** 2 - x_robot ** 2 - x_obst[0] ** 2
    beta_1 = x_robot ** 2 - x_obst[1] ** 2 - q_radii[1] ** 2

    beta_gradient = np.gradient(np.array([beta_0, beta_1]))
    
    v0 = q_radii[0] * (1 - beta_0) / norm(x_robot - x_obst[0])
    v1 = q_radii[1] * (1 + beta_1) / norm(x_robot - x_obst[1])

    gradient_v0 =  v0 * (norm(x_robot - x_obst[0]) / (1 + beta_0) * beta_gradient[0] - (1 / norm(x_robot - x_obst[0])) * (x_robot - x_obst[0]))
    gradient_v1 =  v1 * (norm(x_robot - x_obst[1]) / (1 + beta_1) * beta_gradient[1] - (1 / norm(x_robot - x_obst[1])) * (x_robot - x_obst[1]))

    gamma_d = norm(x_robot - goal[:1]) ** 2

    print(f'beta gr: {beta_gradient}')

    beta_dash_0 = beta_1
    beta_dash_1 = beta_0

    gradient_gamma_d_beta_dash = np.gradient(np.array([gamma_d * beta_dash_0, gamma_d * beta_dash_1]))

    sigma_0 = gamma_d * beta_dash_0 / (gamma_d * beta_dash_0 + lam * beta_0)
    sigma_1 = gamma_d * beta_dash_1 / (gamma_d * beta_dash_1 + lam * beta_1)

    sigma_d = 1 - (sigma_0 + sigma_1)

    gradient_sigma_0 = (lam/((gamma_d*beta_dash_0+lam*beta_0)**2))*(beta_0 * gradient_gamma_d_beta_dash[0] - gamma_d*beta_dash_0*beta_gradient[0])
    gradient_sigma_1 = (lam/((gamma_d*beta_dash_1+lam*beta_1)**2))*(beta_1 * gradient_gamma_d_beta_dash[1] - gamma_d*beta_dash_1*beta_gradient[1])

    print(f'gammad gradient beta dash: {gradient_gamma_d_beta_dash}')
    
    
    J = (sigma_0 * v0 * np.eye(2) + sigma_0 * (x_robot - x_obst[0]) * gradient_v0.T + (v0 - 1) * (x_robot - x_obst[0]) * gradient_sigma_0.T +
         sigma_1 * v1 * np.eye(2) + sigma_1 * (x_robot - x_obst[1]) * gradient_v1.T + (v1 - 1) * (x_robot - x_obst[1]) * gradient_sigma_1.T) + sigma_d * np.eye(2)

    return J


def main():
    # Constants and Settings
    Ts = 0.01 # Update simulation every 10ms
    t_max = 10.0 # total simulation duration in seconds
    # Set initial state
    IS_SHOWING_2DVISUALIZATION = True
    MAX_ANGULAR_VEL = 2.84
    MAX_LINEAR_VEL = 0.5

    field_x = (-2.5, 2.5)
    field_y = (-2.5, 2.5)

    Kp = 1
    l = 0.06
    Kappa = 1
    gamma = 0.1
    lam = 100
    q_t0 = np.array([.0, 1.0])
    r_t0 = 0.5
    t = 0.0
    qi = q_t0
    q = np.array([2, 2])
    r = r_t0
    r0_t0 = 5.0
    r0value = r0_t0
    q0 = np.array([0., 0.])

    x = q
    theta_robot = 0.0
    theta_robot_bw = theta_robot
    xi = qi
    qiF = [q0, qi]
    rho_i = [r0value, r]
    x_obstacleCount = 1
    M = x_obstacleCount + 1
    x_g = np.array([-2, -1, 0.0])
    q_g = x_g


    ux = Kp * (x_g[0] - x[0])
    uy = Kp * (x_g[1] - x[1])

    Jacobian = numerical_jacobian(x, qiF, rho_i, x_g, lam)
    J11 = Jacobian[0][0]   # dF1/dx
    J12 = Jacobian[0][1]   # dF1/dy
    J21 = Jacobian[1][0]   # dF2/dx
    J22 = Jacobian[1][1]   # dF2/dy

    print(f'GOAL P:     {np.array([J11*x_g[0]+J12*x_g[1], J21*x_g[0]+J22*x_g[1]])}')


    #J = calculateJacobian(Fsym)
    #F = diffeomorphismF(M, x, xi, x_g, rho_i, qiF, q_g)

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional', 1)
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(x_g)
        x_values = np.linspace(-5, 5, 100)
        y_values = np.linspace(-5, 5, 100)
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        xc1 = q_t0[0]
        yc1 = q_t0[1]
        sim_visualizer.ax.add_patch(plt.Circle(([xi[0], xi[1]]), r_t0))
        sim_visualizer_bw = sim_mobile_robot( 'omnidirectional', 2)
        sim_visualizer_bw.set_field( [-7.5, 7.5], [-7.5, 7.5]) # set plot area
        sim_visualizer_bw.show_goal(q_g)
        bw_x_obst = plt.Circle((q_t0), r_t0, color='r', fill=False)
        bw_safe_set = plt.Circle((q0), r0_t0, color='b', fill=False)
        sim_visualizer_bw.ax.add_patch(bw_x_obst)
        sim_visualizer_bw.ax.add_patch(bw_safe_set)

    state_history = np.zeros((100000, 2))
    state_history_q = np.zeros((100000, 2))

    step = 0
    while t < t_max and (sqrt((x[0]-x_g[0])**2 + (x[1]-x_g[1])**2) > 0.3):


        Jacobian = numerical_jacobian(x, [q0, xi], rho_i, x_g, lam)
        J11 = Jacobian[0][0]   # dF1/dx
        J12 = Jacobian[0][1]   # dF1/dy
        J21 = Jacobian[1][0]   # dF2/dx
        J22 = Jacobian[1][1]   # dF2/dy

        q_dot = np.array([J11*ux+J12*uy, J21*ux+J22*uy]).T

        # 5

        u_hat_q = Kp*(q_t0-qi)
        u_hat_r = Kp*(r_t0-r)
        u_hat_r0 = Kp*(r0_t0-r0value)
        u_hat = matrix([u_hat_q[0], u_hat_q[1], u_hat_r, u_hat_r0])

        # 6

        # C1 + C0
        # #hi = (np.linalg.norm(qi-q))**2 - r**2
        # hi = sqrt((qi[0]-q[0])**2 + (qi[1]-q[1])**2)**2 - r**2
        # #h0 = r0value**2 - (np.linalg.norm(q0-q))**2
        # h0 = r0value**2 - (sqrt((q0[0]-q[0])**2 + (q0[1]-q[1])**2))**2

        # AC1 = matrix([[-2*(qi[0]-q[0]), -2*(qi[1]-q[1]), 2*r]]).T
        # #print(AC1)
        # AC0 = -2*r0value
        # bC1 = float(-2*(qi-q) @ q_dot + gamma * hi)
        # bC0 = float(2*(q0-q) @ q_dot + gamma * h0)

        # # C2
        # # hij = np.linalg.norm(qi-qj)**2 - (ri-rj)**2
        # # AC2 = matrix([[-2*(qi[0]-qj[0]), -2*(qi[1]-qj[1]), 2*(qi[0]-qj[0]), 2*(qi[1]-qj[1]), 2*(ri+rj), 2*(ri+rj)]]).T
        # # bC2 = gamma*hij

        # # C3
        # hi0 = (r0value-r)**2 - np.linalg.norm(qi-q0)**2
        # hi0 = (r0value-r)**2 - sqrt((qi[0]-q0[0])**2 + (qi[1]-q0[1])**2)**2
        # AC3 = matrix([[2*(qi[0]-q0[0]), 2*(qi[1]-q0[1]), 2*(r0value-r), -2*(r0value-r)]])
        # bC3 = float(gamma*hi0)

        # Q = matrix([[2.0, 0.0, 0.0, 0.0,],
        #             [0.0, 2.0, 0.0, 0.0],
        #             [0.0, 0.0, 2.0*Kappa, 0.0],
        #             [0.0, 0.0, 0.0, 2.0*Kappa]])
                
        # c = matrix([-2*u_hat[0], -2*u_hat[1], -2*u_hat[2], -2*u_hat[3]])

        # # u = [uqx uqy ur ur0]
        # G = matrix([[AC1[0], AC1[1], AC1[2], 0.0],
        #             [0.0, 0.0, 0.0, AC0],
        #             [AC3[0], AC3[1], AC3[2], AC3[3]]]).T
        
        # h = matrix([bC1, bC0, bC3])
        
        # solvers.options['show_progress'] = False
        # sol = solvers.qp(Q, c, G, h, verbose=False)

        # u_star_qx = sol['x'][0]
        # u_star_qy = sol['x'][1]
        # u_star_r = sol['x'][2]
        # u_star_r0 = sol['x'][3]

        # 7
        
        # qi[0] += u_star_qx*Ts
        # qi[1] += u_star_qy*Ts
        # r += u_star_r*Ts
        # r0value += u_star_r0*Ts
        # rho_i = [r0value, r]

        t += Ts

        # 8
        jacobian = numerical_jacobian(x, [q0, xi], rho_i, x_g, lam)
        inv_jacobian = np.linalg.inv( jacobian )

        x_dot = inv_jacobian @ q_dot

        ux = x_dot[0]
        uy = x_dot[1]

        # print(f'ux:  {ux}')
        # print(f'uy:  {uy}')
        print(f'q_dotx:  {q_dot[0]}')
        print(f'q_doty:  {q_dot[1]}')


        control_input = np.array([x_dot[0], x_dot[1]])
        control_input_bw = np.array([q_dot[0], q_dot[1]])
        # control_input[0] = min(max(control_input[0], -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
        # control_input[1] = min(max(control_input[1], -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
        # control_input_bw[0] = min(max(control_input_bw[0], -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
        # control_input_bw[1] = min(max(control_input_bw[1], -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
        # 9
  

        # Update bw
        x = x + Ts*control_input
        q = q + Ts*control_input_bw
        x = x.astype(np.float64)
        q = q.astype(np.float64)

        state_history[step] = x
        state_history_q[step] = q

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp(t)
            sim_visualizer.update_goal( x_g )
            sim_visualizer.fig.canvas.draw()  
            sim_visualizer.update_trajectory( state_history[:step+1] ) # up to the latest data
            sim_visualizer_bw.update_time_stamp(t)
            sim_visualizer_bw.update_goal( q_g )
            sim_visualizer_bw.update_trajectory( state_history_q[:step+1] ) # up to the latest data
            bw_x_obst.set_center(qi)
            bw_x_obst.set_radius(r)
            bw_safe_set.set_radius(r0value)
            sim_visualizer_bw.fig.canvas.draw()
            plt.pause(0.000001)
        step += 1

    plt.show()
main()
