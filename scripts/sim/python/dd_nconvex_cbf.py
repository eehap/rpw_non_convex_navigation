import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
import sys
sys.path.append('/Users/eehaap/robotics_project_work/rpw_non_convex_navigation')
from FunMoRo_control.library.visualize_mobile_robot import sim_mobile_robot


# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 25.0 # total simulation duration in seconds
# Set initial state
init_state = np.array([-2., -.5, 0.]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2.5, 2.5)
R_si = 0.51

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
    l = 0.2
    goal_threshold = 0.05

    x += l * np.cos(theta)
    y += l * np.sin(theta)

    err_goal = np.sqrt((x_d - x) ** 2 + (y_d - y) ** 2)

    k_g = (0.5 * (1 - np.e ** (-beta * err_goal)) / err_goal)
    k_c = 1

    ux_gtg = k_c * (x_d - x)
    uy_gtg = k_c * (y_d - y)

    m_1 = np.array([ [1.0, 0.0], [0.0, 1/l] ])
    m_2 = np.array([ [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)] ]).T
    
    # QP parameters
    Q = 2 * matrix(np.eye(2))

    h_o1 = (np.linalg.norm([x - 0, y - 0])) ** 2 - (R_si ** 2)

    h_o1_d_x = 2 * (x - 0)
    h_o1_d_y = 2 * (y - 0)

    G = matrix([-h_o1_d_x, -h_o1_d_y]).T

    h = matrix(gamma * (h_o1 ** 3))

    c = matrix([-2 * ux_gtg, -2 * uy_gtg])
    
    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, c, G, h, verbose=False)

    ux = sol['x'][0]
    uy = sol['x'][1]

    uc = m_1 @ m_2 @ np.array([ux, uy]).T
    vx = uc[0]
    wz = uc[1]

    # initial numpy array for [ux, uy]
    current_input = np.array([0., 0.]) 
    # Compute the control input
    current_input[0] = vx
    current_input[1] = wz

    return current_input


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([2, 0., 0.]) # numpy array for goal / the desired [px, py, theta]
    obstacles = np.array(([[0.0, 0.0]])) # obstacle center point


    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 2) ) # for [vlin, omega] vs iteration time
    h_function_history = np.zeros( (sim_iter, 1))


    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        # sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)


    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input = compute_control_input(desired_state, robot_state, current_time, obstacles)
        #------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
            sim_visualizer.ax.add_patch(plt.Circle( obstacles[0], 0.3, color='r'))
            sim_visualizer.ax.add_patch(plt.Circle( obstacles[0], R_si, color='r', fill=False))
                
        
        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of UNICYCLE model
        theta = robot_state[2]
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        robot_state = robot_state + Ts*(B @ current_input) # will be used in the next iteration
        robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, h_function_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, h_function_history = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='vx [m/s]')
    ax.plot(t, input_history[:,1], label='omega [rad/s]')
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    ax.plot(t, state_history[:,0], label='px [m]')
    ax.plot(t, state_history[:,1], label='py [m]')
    ax.plot(t, state_history[:,2], label='theta [rad]')
    ax.plot(t, goal_history[:,0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:,1], ':', label='goal py [m]')
    ax.plot(t, goal_history[:,2], ':', label='goal theta [rad]')
    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, h_function_history[:, 0], label='h')
    ax.set(xlabel="t [s]", ylabel="h(x) [m^2]")
    plt.legend()
    plt.grid()

    plt.show()
