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
init_state = np.array([-4., 0.0, 0.0]) # px, py, theta
desired_state = np.array([2, 0., 0.]) # numpy array for goal / the desired [px, py, theta]

obst = np.array(([[0.0, 0.0]])) # obstacle center point
obst_bw_pos_t0 = obst.copy()
obst_bw_r_t0 = np.array([1.0])

IS_SHOWING_2DVISUALIZATION = True
MAX_ANGULAR_VEL = 2.84
MAX_LINEAR_VEL = 0.22

# Define Field size for plotting (should be in tuple)
field_x = (-5, 5)
field_y = (-5, 5)

# Safe set
safe_set_radius_t0 = 5.0
safe_set_center = np.array([0.0, 0.0])


# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, current_time, obst_bw, obst_bw_r, safe_set_r):
    x_d = desired_state[0]
    y_d = desired_state[1]

    x_robot = robot_state[0]
    y_robot = robot_state[1]
    theta_robot = robot_state[2]

    l = 0.06
    goal_threshold = 0.05

    x_robot += l * np.cos(theta_robot)
    y_robot += l * np.sin(theta_robot)

    err_goal = np.sqrt((x_d - x_robot) ** 2 + (y_d - y_robot) ** 2)

    # QP vars
    Kappa = 1.0
    gamma = 1.0

    # Go to goal controller
    k_c = 1
    ux_gtg = k_c * (x_d - x_robot)
    uy_gtg = k_c * (y_d - y_robot)
    
    # Obst nominal control
    Kp = 1.0
    u_hat_q = Kp*(obst_bw_pos_t0 - obst_bw)
    u_hat_r = Kp*(obst_bw_r_t0 - obst_bw_r)
    u_hat_r0 = Kp*(safe_set_radius_t0 - safe_set_r)
    u_hat = matrix([u_hat_q[0], u_hat_q[1], u_hat_r, u_hat_r0])

     # QP parameters


    # Robot control (real world)

    u = np.array((1,2))
    vx = u[0] * np.cos(theta_robot) + u[1] * np.sin(theta_robot)
    wz = (- u[0] * np.sin(theta_robot) + u[1] * np.cos(theta_robot)) / l

    # Confine cmd values to upper and lower bounds
    vx = min(max(vx, -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
    wz = min(max(wz, -MAX_ANGULAR_VEL), MAX_ANGULAR_VEL)

    if err_goal < goal_threshold:
        vx = 0
        wz = 0

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
    obst_bw_pos = obst_bw_pos_t0.copy()
    obst_bw_r = obst_bw_r_t0.copy()
    safe_set_r = safe_set_radius_t0
    
    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 2) ) # for [vlin, omega] vs iteration time


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
        current_input = compute_control_input(robot_state, current_time, obst_bw_pos, obst_bw_r, safe_set_r)
        #------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
        
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
    return state_history, input_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, input_history = simulate_control()


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
