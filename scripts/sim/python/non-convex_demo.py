import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt
from sympy import *
from cvxopt import solvers, matrix
import time
#from torch.autograd.functional import jacobian
from library.visualize_mobile_robot import sim_mobile_robot
from r_function import calculate_r
import json


def calculateTheta(x_robot, x_obstacle):
    theta = atan2(x_robot[1] - x_obstacle[1], x_robot[0] - x_obstacle[0])
    if theta > 180:
        dt = theta - 180
        theta = 180 - dt
    return theta


def jacobianOriginal(r_obstacle, r_bw, lam):
    # Robot x and y coordinates
    x_robot_x, x_robot_y = symbols('x_robot_x x_robot_y')
    # Obstacle x and y coordinates
    x_obstacle_x, x_obstacle_y, q_obstacle_x, q_obstacle_y = symbols('x_obstacle_x x_obstacle_y q_obstacle_x q_obstacle_y')
    # Ball world x and y coordinates
    x_bw_x, x_bw_y, q_bw_x, q_bw_y = symbols('x_bw_x x_bw_y q_bw_x q_bw_y')
    # Goal point
    x_goal_x, x_goal_y, q_goal_x, q_goal_y = symbols('x_goal_x x_goal_y q_goal_x q_goal_y')
    # Transformed radii
    rho_bw_sym, rho_obstacle_sym = symbols('rho_bw_sym rho_obstacle_sym')

    theta0 = atan2(x_robot_y-x_bw_y, x_robot_x-x_bw_x)
    theta1 = atan2(x_robot_y-x_obstacle_y, x_robot_x-x_obstacle_x)
    beta0 = rho_bw_sym**2 - (sqrt((x_robot_x-x_bw_x)**2 + (x_robot_y-x_bw_y)**2))**2
    beta1 = (sqrt((x_robot_x-x_obstacle_x)**2 + (x_robot_y-x_obstacle_y)**2))**2 - rho_obstacle_sym**2
    beta_dash0 = beta1
    beta_dash1 = beta0
    gamma_g = (sqrt((x_robot_x-x_goal_x)**2 + (x_robot_y-x_goal_y)**2))**2
    sigma0 = (gamma_g*beta_dash0)/(gamma_g*beta_dash0+lam*beta0)
    sigma1 = (gamma_g*beta_dash1)/(gamma_g*beta_dash1+lam*beta1)
    sigma_g = 1 - (sigma0+sigma1)
    f0 = (sqrt((x_robot_x-x_bw_x)**2 + (x_robot_y-x_bw_y)**2))/(r_bw) * np.array([cos(theta0), sin(theta0)]).T
    f1 = (sqrt((x_robot_x-x_obstacle_x)**2 + (x_robot_y-x_obstacle_y)**2))/(r_obstacle) * np.array([cos(theta1), sin(theta1)]).T
    
    Fx = sigma0*(rho_bw_sym*f0 + np.array([q_bw_x, q_bw_y]))
    Fx += sigma1*(rho_obstacle_sym*f1 + np.array([q_obstacle_x, q_obstacle_y]))
    Fx += sigma_g*(np.array([x_robot_x, x_robot_y]) - np.array([x_goal_x, x_goal_y]) + np.array([q_goal_x, q_goal_y]))
    
    J11 = diff(Fx[0],x_robot_x)
    J12 = diff(Fx[0],x_robot_y)
    J21 = diff(Fx[1],x_robot_x)
    J22 = diff(Fx[1],x_robot_y)
    J = [J11, J12, J21, J22]
    return J

def diffeomorphism(x_robot, x_bw, r_bw, x_obstacle, r_obstacle, q_obstacle, rho_bw, rho_obstacle, x_goal, q_goal, lam):
    theta0 = atan2(x_robot[1]-x_bw[1], x_robot[0]-x_bw[0])
    theta1 = atan2(x_robot[1]-x_obstacle[1], x_robot[0]-x_obstacle[0])
    beta0 = rho_bw**2 - (sqrt((x_robot[0]-x_bw[0])**2 + (x_robot[1]-x_bw[1])**2))**2
    #beta1 = (sqrt((x_robot[0]-x_obstacle[0])**2 + (x_robot[1]-x_obstacle[1])**2))**2 - rho_obstacle**2
    a = 1
    b = 1.1
    beta1 = ((x_robot[0]-x_obstacle[0]-a)**2 + (x_robot[1]-x_obstacle[1])**2) * ((x_robot[0]-x_obstacle[0]+a)**2 + (x_robot[1]-x_obstacle[1])**2) - b**4
    beta_dash0 = beta1
    beta_dash1 = beta0
    gamma_g = (sqrt((x_robot[0]-x_goal[0])**2 + (x_robot[1]-x_goal[1])**2))**2
    sigma0 = (gamma_g*beta_dash0)/(gamma_g*beta_dash0+lam*beta0)
    sigma1 = (gamma_g*beta_dash1)/(gamma_g*beta_dash1+lam*beta1)
    sigma_g = 1 - (sigma0+sigma1)
    f0 = (sqrt((x_robot[0]-x_bw[0])**2 + (x_robot[1]-x_bw[1])**2))/(r_bw) * np.array([cos(theta0), sin(theta0)]).T
    f1 = (sqrt((x_robot[0]-x_obstacle[0])**2 + (x_robot[1]-x_obstacle[1])**2))/(r_obstacle) * np.array([cos(theta1), sin(theta1)]).T
    
    Fx = sigma0*(rho_bw*f0 + np.array([x_bw[0], x_bw[1]]))
    Fx += sigma1*(rho_obstacle*f1 + np.array([q_obstacle[0], q_obstacle[1]]))
    Fx += sigma_g*(np.array([x_robot[0], x_robot[1]]) - np.array([x_goal[0], x_goal[1]]) + np.array([q_goal[0], q_goal[1]]))
    
    #print(Fx)
    return Fx


def main():
    # Constants and Settings
    Ts = 0.01 # Update simulation every 10ms
    t_max = 30.0 # total simulation duration in seconds
    IS_SHOWING_2DVISUALIZATION = True
    MAX_ANGULAR_VEL = 2.84
    MAX_LINEAR_VEL = 0.5
    H = 10e-6

    field_x = (-5, 5)
    field_y = (-5, 5)
    field_x_bw = (-7.5, 7.5)
    field_y_bw = (-7.5, 7.5)

    # Set initial state
    Kp = 1
    K_gtg = 1
    Kappa = 1
    gamma = 1
    lam = 1000
    q_obstacle_t0 = np.array([0.5, 0.5])
    t = 0.0
    x_obstacle = np.array([0.5, 0.5])
    x_obstacle_edge = np.array([0.5, 0.5])

    q_obstacle = np.array([0.5, 0.5])
    x_bw = np.array([0.0, 0.0])
    q_bw = np.array([0.0, 0.0])
    x_robot = np.array([1.5, 1.5])
    q_robot = np.array([1.5, 1.5])
    theta_for_r_calculation = calculateTheta(x_robot, x_obstacle)
    r_obstacle, th = calculate_r(x_robot, x_obstacle)
    print(f'r_obstacle: {r_obstacle}')
    rho_obstacle = 1.5
    rho_obstacle_t0 = 1.5
    rho_bw_t0 = 5.0
    r_bw = 5.0
    rho_bw = 5.0

    theta_robot = 0.0
    theta_robot_bw = 0.0
    x_goal = np.array([-2., -1., 0.0])
    q_goal = np.array([-2., -1., 0.0])

    F = diffeomorphism(x_robot, x_bw, r_bw, x_obstacle, r_obstacle, q_obstacle, rho_bw, rho_obstacle, x_goal, q_goal, lam)
    q_robot[0] = F[0]
    q_robot[1] = F[1]

    ux = K_gtg * (x_goal[0] - x_robot[0])
    uy = K_gtg * (x_goal[1] - x_robot[1])

    # Robot x and y coordinates
    x_robot_x, x_robot_y = symbols('x_robot_x x_robot_y')
    # Obstacle x and y coordinates
    x_obstacle_x, x_obstacle_y, q_obstacle_x, q_obstacle_y = symbols('x_obstacle_x x_obstacle_y q_obstacle_x q_obstacle_y')
    # Ball world x and y coordinates
    x_bw_x, x_bw_y, q_bw_x, q_bw_y = symbols('x_bw_x x_bw_y q_bw_x q_bw_y')
    # Goal point
    x_goal_x, x_goal_y, q_goal_x, q_goal_y = symbols('x_goal_x x_goal_y q_goal_x q_goal_y')
    # Transformed radii
    rho_bw_sym, rho_obstacle_sym = symbols('rho_bw_sym rho_obstacle_sym')

    jacobian = jacobianOriginal(r_obstacle, r_bw, lam)
    J11 = jacobian[0]   # dF1/dx
    J12 = jacobian[1]   # dF1/dy
    J21 = jacobian[2]   # dF2/dx
    J22 = jacobian[3]   # dF2/dy
    J11_lambd = lambdify([x_robot_x, x_robot_y, x_obstacle_x, x_obstacle_y, q_obstacle_x, q_obstacle_y,
                          x_bw_x, x_bw_y, q_bw_x, q_bw_y, x_goal_x, x_goal_y, q_goal_x, q_goal_y, rho_bw_sym, rho_obstacle_sym], J11, "numpy")
    J12_lambd = lambdify([x_robot_x, x_robot_y, x_obstacle_x, x_obstacle_y, q_obstacle_x, q_obstacle_y,
                          x_bw_x, x_bw_y, q_bw_x, q_bw_y, x_goal_x, x_goal_y, q_goal_x, q_goal_y, rho_bw_sym, rho_obstacle_sym], J12, "numpy")
    J21_lambd = lambdify([x_robot_x, x_robot_y, x_obstacle_x, x_obstacle_y, q_obstacle_x, q_obstacle_y,
                          x_bw_x, x_bw_y, q_bw_x, q_bw_y, x_goal_x, x_goal_y, q_goal_x, q_goal_y, rho_bw_sym, rho_obstacle_sym], J21, "numpy")
    J22_lambd = lambdify([x_robot_x, x_robot_y, x_obstacle_x, x_obstacle_y, q_obstacle_x, q_obstacle_y,
                          x_bw_x, x_bw_y, q_bw_x, q_bw_y, x_goal_x, x_goal_y, q_goal_x, q_goal_y, rho_bw_sym, rho_obstacle_sym], J22, "numpy")

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
            sim_visualizer = sim_mobile_robot( 'omnidirectional', 1)
            sim_visualizer.set_field( field_x, field_y ) # set plot area
            sim_visualizer.show_goal(x_goal)
            #sim_visualizer.ax.add_patch(plt.Circle(([x_obstacle[0], x_obstacle[1]]), rho_obstacle_t0 - 0.2))
            x = np.linspace(-10, 10, 1000)
            y = np.linspace(-10, 10, 1000)
            X, Y = np.meshgrid(x, y)
            a = 1.0
            b = 1.1
            XY = ((X - x_obstacle[0] - a) ** 2 + ((Y - x_obstacle[1]) ** 2)) * (((X - x_obstacle[0] + a) ** 2) + ((Y - x_obstacle[1]) ** 2)) - (b ** 4)
            sim_visualizer.ax.contour(X, Y, XY, levels=[0])

            # Ball world sim
            sim_visualizer_bw = sim_mobile_robot( 'omnidirectional', 2)
            sim_visualizer_bw.set_field( field_x_bw, field_y_bw) # set plot area
            sim_visualizer_bw.show_goal(q_goal)
            bw_x_obst = plt.Circle((q_obstacle_t0), rho_obstacle_t0, color='r', fill=False)
            bw_safe_set = plt.Circle((q_bw), rho_bw_t0, color='b', fill=False)
            sim_visualizer_bw.ax.add_patch(bw_x_obst)
            sim_visualizer_bw.ax.add_patch(bw_safe_set)

    state_history = np.zeros((100000, 2))
    state_history_q = np.zeros((100000, 2))

    k = 0
    while t < t_max:
        theta_for_r_calculation = calculateTheta(x_robot, x_obstacle)
        r_obstacle, th = calculate_r(x_robot, x_obstacle)
        #print(f'r_obstacle, theta: {r_obstacle, np.degrees(theta)}')
        #x_obstacle_edge[0] = x_obstacle[0] + r_obstacle*np.cos(theta)
        #x_obstacle_edge[1] = x_obstacle[1] + r_obstacle*np.sin(theta)
        #print(f'x_obstacle_edge: {x_obstacle_edge}')

        # 3 
        print('kierros', k)

        ux = K_gtg * (x_goal[0] - x_robot[0])
        uy = K_gtg * (x_goal[1] - x_robot[1])
        x_dot = np.array([ux, uy])
        norm_vel = np.hypot(x_dot[0], x_dot[1])
        if norm_vel > MAX_LINEAR_VEL: x_dot = MAX_LINEAR_VEL* x_dot / norm_vel

        # 4

        # J11s = J11_lambd(x_robot[0], x_robot[1], x_obstacle[0], x_obstacle[1], q_obstacle[0], q_obstacle[1], x_bw[0], x_bw[1],
        #                     q_bw[0], q_bw[1], x_goal[0], x_goal[1], q_goal[0], q_goal[1], rho_bw, rho_obstacle)
        # J12s = J12_lambd(x_robot[0], x_robot[1], x_obstacle[0], x_obstacle[1], q_obstacle[0], q_obstacle[1], x_bw[0], x_bw[1],
        #                     q_bw[0], q_bw[1], x_goal[0], x_goal[1], q_goal[0], q_goal[1], rho_bw, rho_obstacle)
        # J21s = J21_lambd(x_robot[0], x_robot[1], x_obstacle[0], x_obstacle[1], q_obstacle[0], q_obstacle[1], x_bw[0], x_bw[1],
        #                     q_bw[0], q_bw[1], x_goal[0], x_goal[1], q_goal[0], q_goal[1], rho_bw, rho_obstacle)
        # J22s = J22_lambd(x_robot[0], x_robot[1], x_obstacle[0], x_obstacle[1], q_obstacle[0], q_obstacle[1], x_bw[0], x_bw[1],
        #                     q_bw[0], q_bw[1], x_goal[0], x_goal[1], q_goal[0], q_goal[1], rho_bw, rho_obstacle)
        
        x_h = np.array([x_robot[0]+H, x_robot[1]])
        y_h = np.array([x_robot[0], x_robot[1]+H])
        Fx = diffeomorphism(x_h, x_bw, r_bw, x_obstacle, r_obstacle, q_obstacle, rho_bw, rho_obstacle, x_goal, q_goal, lam)
        Fy = diffeomorphism(y_h, x_bw, r_bw, x_obstacle, r_obstacle, q_obstacle, rho_bw, rho_obstacle, x_goal, q_goal, lam)
        F = diffeomorphism(x_robot, x_bw, r_bw, x_obstacle, r_obstacle, q_obstacle, rho_bw, rho_obstacle, x_goal, q_goal, lam)
        J11s = float((Fx[0]-F[0])/H)
        J12s = float((Fy[0]-F[0])/H)
        J21s = float((Fx[1]-F[1])/H)
        J22s = float((Fy[1]-F[1])/H)
        
        q_dot = np.array([J11s*x_dot[0]+J12s*x_dot[1], J21s*x_dot[0]+J22s*x_dot[1]])

        # 5

        u_hat_q_obstacle = Kp*(q_obstacle_t0-q_obstacle)
        u_hat_rho_obstacle = Kp*(rho_obstacle_t0-rho_obstacle)

        u_hat_rho_bw = Kp*(rho_bw_t0-rho_bw)

        # 6

        # C1:
        AC1_0 = -2*rho_bw
        AC1_1 = matrix([[-2*(q_obstacle[0]-q_robot[0]), -2*(q_obstacle[1]-q_robot[1]), 2*rho_obstacle]])
        h1 = norm(q_obstacle-q_robot)**2 - rho_obstacle**2
        h0 = rho_bw**2 - norm(q_bw-q_robot)**2
        bC1_1 = -2*(q_obstacle-q_robot) @ q_dot + gamma * h1
        bC1_0 = 2*(q_bw - q_robot) @ q_dot + gamma * h0

        # C3:
        AC3 = matrix([[2*(q_obstacle[0]-q_bw[0]), 2*(q_obstacle[1]-q_bw[1]), 2*(rho_bw-rho_obstacle), -2*(rho_bw-rho_obstacle)]])
        hi0 = (rho_bw-rho_obstacle)**2 - norm(q_obstacle-q_bw)**2
        bC3 = gamma * hi0

        Q = matrix([[2.0, 0.0, 0.0, 0.0,],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0*Kappa, 0.0],
                    [0.0, 0.0, 0.0, 2.0*Kappa]])
        c = matrix([-2*u_hat_q_obstacle[0], -2*u_hat_q_obstacle[1], -2*u_hat_rho_obstacle, -2*u_hat_rho_bw])

        G = matrix([[AC1_1[0], AC1_1[1], AC1_1[2], 0.0],
                    [0.0, 0.0, 0.0, AC1_0],
                    [AC3[0], AC3[1], AC3[2], AC3[3]]]).T
        
        h = matrix([bC1_1, bC1_0, bC3])
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, c, G, h, verbose=False)

        u_star_q_obstacle_x = sol['x'][0]
        u_star_q_obstacle_y = sol['x'][1]
        u_star_rho_obstacle = sol['x'][2]
        u_star_rho_bw = sol['x'][3]

        # 7

        q_obstacle[0] += u_star_q_obstacle_x*Ts
        q_obstacle[1] += u_star_q_obstacle_y*Ts

        print(q_obstacle)
        rho_obstacle += u_star_rho_obstacle*Ts  
        print(rho_obstacle)
        rho_bw += u_star_rho_bw*Ts
        t += Ts

        # 8

        # J11s = J11_lambd(q_robot[0], x_robot[1], x_obstacle[0], x_obstacle[1], q_obstacle[0], q_obstacle[1], x_bw[0], x_bw[1],
        #                  q_bw[0], q_bw[1], x_goal[0], x_goal[1], q_goal[0], q_goal[1], rho_bw, rho_obstacle)
        # J12s = J12_lambd(q_robot[0], x_robot[1], x_obstacle[0], x_obstacle[1], q_obstacle[0], q_obstacle[1], x_bw[0], x_bw[1],
        #                  q_bw[0], q_bw[1], x_goal[0], x_goal[1], q_goal[0], q_goal[1], rho_bw, rho_obstacle)
        # J21s = J21_lambd(q_robot[0], x_robot[1], x_obstacle[0], x_obstacle[1], q_obstacle[0], q_obstacle[1], x_bw[0], x_bw[1],
        #                  q_bw[0], q_bw[1], x_goal[0], x_goal[1], q_goal[0], q_goal[1], rho_bw, rho_obstacle)
        # J22s = J22_lambd(q_robot[0], x_robot[1], x_obstacle[0], x_obstacle[1], q_obstacle[0], q_obstacle[1], x_bw[0], x_bw[1],
        #                  q_bw[0], q_bw[1], x_goal[0], x_goal[1], q_goal[0], q_goal[1], rho_bw, rho_obstacle)

        qx_h = np.array([q_robot[0]+H, q_robot[1]])
        qy_h = np.array([q_robot[0], q_robot[1]+H])
        Fx = diffeomorphism(qx_h, x_bw, r_bw, x_obstacle, r_obstacle, q_obstacle, rho_bw, rho_obstacle, x_goal, q_goal, lam)
        Fy = diffeomorphism(qy_h, x_bw, r_bw, x_obstacle, r_obstacle, q_obstacle, rho_bw, rho_obstacle, x_goal, q_goal, lam)
        F = diffeomorphism(q_robot, x_bw, r_bw, x_obstacle, r_obstacle, q_obstacle, rho_bw, rho_obstacle, x_goal, q_goal, lam)
        J11s = float((Fx[0]-F[0])/H)
        J12s = float((Fy[0]-F[0])/H)
        J21s = float((Fx[1]-F[1])/H)
        J22s = float((Fy[1]-F[1])/H)
        
        jacobian = np.array([[J11s, J12s], [J21s, J22s]], dtype=float)
        inv_jacobian = np.linalg.inv(jacobian)
        x_dot = inv_jacobian @ q_dot

        #print(f'x_dot: {x_dot}')
        #print(f'gtg x: {ux}, gtg y: {uy}')

        # 9 (Incomplete?)
        norm_vel = np.hypot(x_dot[0], x_dot[1])
        if norm_vel > MAX_LINEAR_VEL: x_dot = MAX_LINEAR_VEL* x_dot / norm_vel


        ux = x_dot[0]
        uy = x_dot[1]

        # print(f'ux:  {ux}')
        # print(f'uy:  {uy}')

        control_input = np.array([ux, uy])
        control_input_bw = np.array([q_dot[0], q_dot[1]])

        x_robot = x_robot + Ts*control_input
        F = diffeomorphism(x_robot, x_bw, r_bw, x_obstacle, r_obstacle, q_obstacle, rho_bw, rho_obstacle, x_goal, q_goal, lam)
        q_robot[0] = F[0]
        q_robot[1] = F[1]
    

        state_history[k] = x_robot
        state_history_q[k] = q_robot

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp(t)
            sim_visualizer.update_goal(x_goal)
            sim_visualizer.fig.canvas.draw()  
            sim_visualizer.update_trajectory(state_history[:k+1]) # up to the latest data

            sim_visualizer_bw.update_time_stamp(t)
            sim_visualizer_bw.update_goal(q_goal)
            sim_visualizer_bw.update_trajectory(state_history_q[:k+1]) # up to the latest data
            bw_x_obst.set_center(q_obstacle)
            bw_x_obst.set_radius(rho_obstacle)
            bw_safe_set.set_radius(rho_bw)
            sim_visualizer_bw.fig.canvas.draw()
            plt.pause(0.000001)  
        
        k += 1

    
    plt.show()

main()
