import numpy as np

import cvxopt
from cvxopt import matrix, solvers

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


MAX_ANGULAR_VEL = 2.84
MAX_LINEAR_VEL = 0.22

class NonConvexController(Node):
    def __init__(self):
        super.__init__('nonconvex_controller')
        ns = 'ncvx'


        self.cmd_vel_pub = self.create_publisher(Twist, f'{ns}/cmd_vel', 10)

        self.odom_sub = self.create_subscription(Odometry, f'{ns}/odom', 10, self.odom_callback)

        self.x_robot = 0.0
        self.y_robot = 0.0
        self.theta_robot = 0.0

        self.vx = 0.0
        self.wz = 0.0
        

    def publish_cmd(self, cmd):
        cmd = Twist()

        # Confine cmd values to upper and lower bounds
        cmd.linear.x = min(max(self.vx, -MAX_LINEAR_VEL), MAX_LINEAR_VEL)
        cmd.angular.z = min(max(self.wz, -MAX_ANGULAR_VEL), MAX_ANGULAR_VEL)

        self.cmd_vel_pub.publish(cmd)

    # def odom_callback(self, msg):
    #     msg.pose

    def set_init_pose(self, x, y, theta):
        self.x_robot = x
        self.y_robot = y
        self.theta_robot = theta


    ### Helper functions ###
    def yaw_from_quat(quaternion):
        # Extract quaternion components
        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        yaw = np.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

        # Convert radians to degrees
        return yaw



def main():
    pass

if __name__ == '__main__':
    main()