#!/bin/python3
# from threading import Thread, Lock

import numpy as np

import cvxopt
from cvxopt import matrix, solvers

import rclpy
from rclpy.node import Node
from rclpy.executors import Executor

from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import Float32MultiArray

MAX_ANGULAR_VEL = 2.84
MAX_LINEAR_VEL = 0.22
N_OBSTACLES = 1
CONTROL_RATE = 50

class NonConvexController(Node):
    def __init__(self):
        super.__init__('nonconvex_controller')

        self.tb_index = 0
        self.tb_name = f'tb3_{self.tb_index}'

        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.tb_name}/cmd_vel', 10)

        self.pose_sub = self.create_subscription(Pose2D, f'/{self.tb_name}/pos', 10, self.pose_callback)
        self.goal_sub = self.create_subscription(Float32MultiArray, '/ui/goal', 10, self.new_goal_callback)
        self.k = 0

        # lookahead
        self.l = 0.06

        # Obstacle vars
        self.obstacles = np.zeros((N_OBSTACLES), dtype=float)
        self.obstacles_ballworld = np.zeros((N_OBSTACLES), dtype=float)

        # Goal
        self.x_g = 0.0
        self.y_g = 0.0

        # Robot vars
        self.x_robot = 0.0
        self.y_robot = 0.0
        self.theta_robot = 0.0

        self.vx = 0.0
        self.wz = 0.0

    def control_step(self):
        u = np.zeros((2), dtype=float)







        self.compute_u_from_si(u_si=u)
        self.publish_cmd

    def compute_u_from_si(self, u_si):
        self.vx = u_si[0] * np.cos(self.theta_robot) + u_si[1] * np.sin(self.theta_robot)
        self.wz = (- u_si[0] * np.sin(self.theta_robot) + u_si[1] * np.cos(self.theta_robot)) / self.l 

    def publish_cmd(self):
        cmd = Twist()

        # Confine cmd values to upper and lower bounds
        cmd.linear.x = min(max(self.vx, -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = min(max(self.wz, -MAX_ANGULAR_VEL), MAX_ANGULAR_VEL)

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


def main():
    rclpy.init()
    controller = NonConvexController()
    callback_thread = Executor()
    callback_thread.add_node(controller)
    callback_thread.spin()

    rate = controller.create_rate(CONTROL_RATE)
    k = 0

    while rclpy.ok():
        controller.control_step(k)
        k +=1
        rate.sleep()

if __name__ == '__main__':
    main()