import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
import sys
sys.path.append('/Users/eehaap/robotics_project_work/rpw_non_convex_navigation')
from FunMoRo_control.library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.005 # Update simulation every 10ms
t_max = 25  # total simulation duration in seconds
# Set initial state
init_state = np.array([1, -4, 0.]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-5, 5)
field_y = (-5, 5)


# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, current_time, obstacles):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()

    x_d = desired_state[0]
    y_d = desired_state[1]
    theta_d = desired_state[2]

    x = robot_state[0]
    y = robot_state[1]
    theta = robot_state[2]

    # Go to goal controller
    beta = 5
    gamma = 1
    k_wz = 2
    goal_threshold = 0.05

    err_goal = np.sqrt((x_d - x) ** 2 + (y_d - y) ** 2)

    k_g = (0.5 * (1 - np.e ** (-beta * err_goal)) / err_goal)
    k_s = 2

    u_gtg_x = k_s * (x_d - x)
    u_gtg_y = k_s * (y_d - y)

    # QP parameters
    x_c = obstacles[0]
    y_c =  obstacles[1]
    Q = 2* matrix(np.eye(2))
    h_o1 = (np.linalg.norm([x - x_c, y - y_c]) ** 4) - np.array([ [x - x_c], [y - y_c] ]).T @ np.array([ [10.0, 0.0], [0.0, -1.0] ]) @ np.array([ [x - x_c], [y - y_c] ])
    h_o1_d_x = 4 * (np.linalg.norm([x - x_c, y - y_c]) ** 2) * (x - x_c) - 20 * x + 20 * x_c
    h_o1_d_y = 4 * (np.linalg.norm([x - x_c, y - y_c]) ** 2) * (y - y_c) + 2 * y - 2 * y_c
  

    G = matrix([ [-h_o1_d_x, -h_o1_d_y] ]).T

    print(f'h(x): {h_o1}')
    print(f'G: {G}')

    h = matrix(gamma * ((h_o1) ** 1))

    c = matrix([-2 * u_gtg_x, -2 * u_gtg_y])
    
    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, c, G, h, verbose=False)

    ux = sol['x'][0]
    uy = sol['x'][1]
    wz = 0
 

    # if err_goal < goal_threshold:
    #     u_x = 0
    #     u_y = 0
    #     theta_err = theta_d - theta
    #     theta_err = ( (theta_err + np.pi) % (2*np.pi) ) - np.pi
    #     wz = k_wz * theta_err

    #     if np.abs(theta_err) < goal_threshold:
    #         wz = 0

    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.]) 
    # Compute the control input
    current_input[0] = ux
    current_input[1] = uy
    current_input[2] = wz

    return current_input


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([0, 4., 0.]) # numpy array for goal / the desired [px, py, theta]
    obstacles = np.array(([0, 0]))

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 3) ) # for [vx, vy, omega] vs iteration time
    h_function_history = np.zeros( (sim_iter, 1))
    u_gtg_history = np.zeros ( (sim_iter, 2))

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)
        x = np.linspace(-4, 4, 400)
        y = np.linspace(-4, 4, 400)

        # Create a grid of (x, y) values
        X, Y = np.meshgrid(x, y)

        # Define the center of the obstacle
        x_c = 0.0
        y_c = 0.0

        # Calculate h_o1 for each (x, y) pair
        h_o1 = (np.linalg.norm([X - x_c, Y - y_c], axis=0) ** 4) - (X - x_c) * 10 * (X - x_c) - (Y - y_c) * (Y - y_c)
        sim_visualizer.ax.contourf(X, Y, h_o1, levels=[-1e10, 0], colors='r', alpha=0.2)

    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input  = compute_control_input(desired_state, robot_state, current_time, obstacles)
        #------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input
  
        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
        
        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts*current_input # will be used in the next iteration
        robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, h_function_history, u_gtg_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, h_function_history, u_gtg_history = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='u_x [m/s]')
    ax.plot(t, input_history[:,1], label='u_y [m/s]')
    ax.plot(t, u_gtg_history[:,0], label='u_gtg_x [m/s]')
    ax.plot(t, u_gtg_history[:,1], label='u_gtg_y [m/s]')
    ax.set(xlabel="t [s]", ylabel="control input (γ = 0.2)")
    plt.legend()
    plt.grid()

    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, h_function_history[:, 0], label='ho1')
    ax.set(xlabel="t [s]", ylabel="h(x) [m^2]")
    plt.legend()
    plt.grid()

    plt.show()
