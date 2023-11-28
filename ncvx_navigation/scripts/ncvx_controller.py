#!/bin/python3
# from threading import Thread, Lock

import numpy as np
import cvxpy as cp
import yaml
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.executors import Executor
from rclpy.time import Time

from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Float32MultiArray

MAX_ANGULAR_VEL = 2.84
MAX_LINEAR_VEL = 0.22
N_OBSTACLES = 1
CONTROL_RATE = 50
OBSTACLE_FILE = True

class NonConvexController(Node):
    def __init__(self):
        super.__init__('nonconvex_controller')

        self.tb_index = 0
        self.tb_name = f'tb3_{self.tb_index}'

        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.tb_name}/cmd_vel', 10)

        self.pose_sub = self.create_subscription(Pose2D, f'/{self.tb_name}/pos', 10, self.pose_callback)
        self.goal_sub = self.create_subscription(Float32MultiArray, '/ui/goal', 10, self.new_goal_callback)

        # Obstacle vars
        self.obstacles = np.zeros((N_OBSTACLES, 2), dtype=float)
        self.obst_q_pos_t0 = np.zeros((N_OBSTACLES, 2), dtype=float)
        self.obst_q_r_t0 = np.zeros(N_OBSTACLES + 1, dtype=float)
        self.safe_set_centre = np.zeros((1, 2), dtype=float)        

        if OBSTACLE_FILE:
            self.parse_obstacles(file_path='/config/obstacles.yaml')
        
        self.obst_q_pos = self.obst_q_pos_t0
        self.obst_q_r = self.obst_q_r_t0

        # Goal point
        self.x_g = 0.0
        self.y_g = 0.0

        # State and robot vars
        self.x_robot = np.array((1, 3), dtype=float)
        self.q_robot = np.array((1, 2), dtype=float)

        # lookahead
        self.l = 0.06

        # Control vars

        self.u = np.zeros((2), dtype=float)
        self.Kp = 1
        self.kappa = 0
        self.gamma = 1
        self.dt = 1 / CONTROL_RATE

        # Jacobian of diffeomorphic function
        self.J_Fx = np.zeros((2, 2))
        self.q_dot = np.array((1, 2), dtype=float)

    def control_step(self):

        # Compute q_dot(step)

        # Compute u_hat_q, u_hat_ro
        u_hat_q =  self.Kp * (self.obst_q_pos_t0 - self.obst_q_pos)
        u_hat_ro = self.Kp * (self.obst_q_r_t0 - self.obst_q_r)

        # Compute u_star_q, u_star_ro
        Q = 2 * np.eye(3)
        c = np.array([2 * u_hat_q[0], 2 * u_hat_q[1], 2 * self.kappa * u_hat_ro])

        # QP constraints

        # barrier functions h(x)
        b_c1_i = - 2 * (self.obst_q_pos - self.q_robot).T * self.q_dot + self.gamma * ((np.linalg.norm(self.obst_q_pos - self.q_robot) ** 2) - self.obst_q_r[1:] ** 2)
        b_c1_0 = 2 * (self.safe_set_centre - self.obst_q_pos).T * self.q_dot + self.gamma * (self.obst_q_r[0] ** 2 - (np.linalg.norm(self.safe_set_centre - self.obst_q_pos) ** 2))
        b_c2 = self.gamma * ((np.linalg.norm(self.obst_q_pos[:-1] - self.obst_q_pos[1:]) ** 2) - (self.obst_q_r[1:-1] - self.obst_q_r[2:]) ** 2)
        b_c3 = self.gamma * ((self.obst_q_r[0] - self.obst_q_r[1:]) ** 2 - (np.linalg.norm(self.obst_q_pos - self.safe_set_centre) ** 2))   


        # h(x) derivatives
        A_c1_i = np.array([-2 * (self.obst_q_pos - self.q_robot).T, 2 * self.obst_q_r])
        A_c1_0 = - 2 * self.obst_q_r[0]
        A_c2 = np.array([-2 * (self.obst_q_pos[:-1] - self.obst_q_pos[1:]).T, 2 * (self.obst_q_pos[:-1] - self.obst_q_pos[1:]).T,
                          2 * (self.obst_q_r[1:-1]+self.obst_q_r[2:]), 2 * (self.obst_q_r[1:-1]+self.obst_q_r[2:])])
        A_c3 = np.array([ 2 * (self.obst_q_pos - self.safe_set_centre).T, 2 * (self.obst_q_r[0]) - self.obst_q_r[1:], -2 * (self.obst_q_r[0]) - self.obst_q_r[1:]])
        

        u_ro = cp.Variable((N_OBSTACLES + 1, 1))
        u_q = cp.Variable((N_OBSTACLES, 2))
        problem = cp.Problem(cp.Minimize((np.linalg.norm(u_q - u_hat_q) ** 2 + self.kappa * (np.linalg.norm(u_ro - u_hat_ro) ** 2))), 
                             [A_c1_i @ np.array([u_q, u_ro[1:]]).T <= b_c1_i,
                              A_c1_0 @ np.array(u_ro[0]) <= b_c1_0,
                              A_c2 @ np.array([u_q[:-1], u_q[1:], u_ro[:-1], u_ro[1:]]).T <= b_c2,
                              A_c3 @ np.array(u_q, u_ro[1:], u_ro[0]).T <= b_c3])
        problem.solve()

        # u_star_qx = sol['x'][0]
        # u_star_qy = sol['x'][1]
        # u_star_r = sol['x'][2]
        
        # Update bw obstacle radii and pos
        # self.obst_q_pos[:, 0] += u_star_qx * self.dt
        # self.obst_q_pos[:, 1] += u_star_qy * self.dt
        # self.obst_q_r += u_star_r * self.dt

        # Update J_Fx(step+1) from new positions

        # Compute diff J_Fx^(step+1)(-1) / q * q_dot(step)
        
        self.publish_cmd

    def publish_cmd(self):
        cmd = Twist()

        # Single integrator to diff kinematics
        vx = self.u[0] * np.cos(self.theta_robot) + self.u[1] * np.sin(self.theta_robot)
        wz = (- self.u[0] * np.sin(self.theta_robot) + self.u[1] * np.cos(self.theta_robot)) / self.l 

        # Confine cmd values to upper and lower bounds
        cmd.linear.x = min(max(vx, -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = min(max(wz, -MAX_ANGULAR_VEL), MAX_ANGULAR_VEL)

        self.cmd_vel_pub.publish(cmd)

    def pose_callback(self, msg):
        self.x_robot = msg.x
        self.y_robot = msg.y
        self.theta_robot = msg.theta

    def new_goal_callback(self, msg):
        self.x_g = msg.data[0]
        self.y_g = msg.data[1]

    def set_init_pose(self, x, y, theta):
        self.x_robot = x
        self.y_robot = y
        self.theta_robot = theta

    def parse_obstacles(self, file_path):
        with open(file_path, 'r') as file:
            obstacles = yaml.safe_load(file)
            self.safe_set_centre = [obstacles['boundary'][0]['x'], obstacles['boundary'][0]['y']]
            self.obst_q_r_t0[0] = obstacles['boundary'][0]['r']

            for i in range(len(obstacles)):
                self.obst_q_pos_t0[i][0] = obstacles['obstacles'][i]['x']
                self.obst_q_pos_t0[i][1] = obstacles['obstacles'][i]['y']
                self.obst_q_r_t0[i+1] = obstacles['obstacles'][i]['r']

def main():
    rclpy.init()
    controller = NonConvexController()
    callback_thread = Executor()
    callback_thread.add_node(controller)
    callback_thread.spin()

    rate = controller.create_rate(CONTROL_RATE)
    k = 0

    while rclpy.ok():
        k +=1
        controller.control_step(k)
        rate.sleep()

if __name__ == '__main__':
    main()
