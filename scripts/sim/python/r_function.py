from sympy import *
import math
from star_shape import star_shape
import numpy as np
from bean_shape import bean_shape

init_printing(use_unicode=True)

x, y, xc, yc, r, m, b, h = symbols('x y xc yc r m b h')

def calcuate_r(robot_position, obstacle_position):

    x_robot_pos = robot_position[0]
    y_robot_pos = robot_position[1]

    xc = obstacle_position[0]
    yc = obstacle_position[1]
    # Calculate angle
    x_dif = x_robot_pos - xc
    y_dif = y_robot_pos - yc

    m = y_dif/x_dif
    b = y_robot_pos - (m*x_robot_pos)

    #star_shape_func, points =  star_shape(xc,yc,4,1)
    star_shape_func = bean_shape(xc,yc)
    star_shape_func =  solve(star_shape_func,y)

    intersection_points = []

    # shape can have multiple solutions. Need to go through them
    for solution in star_shape_func:
        h_function = Eq(solution,h)

        line_equation = Eq(y, m*x + b)

        # Substitute y from the line equation into the h function
        intersection_equation = h_function.subs(h, m*x + b)
        # Solve for x to find the x-coordinates of intersection points
        x_values = solve(intersection_equation, x)
        #print('x_values: ' ,x_values)

        # Substitute x values back into the line equation to find y-coordinates
        for x_val in x_values:
            if x_val.is_real:
                intersection_points.append([x_val, line_equation.subs(x, x_val).rhs])

    intersection_points = np.array(intersection_points, dtype=np.float32)
    #print("Intersection points:", intersection_points)

    # Solve for closest intersection point and calculate r
    robot_pos = np.array([x_robot_pos, y_robot_pos])
    obstacle_pos = np.array([xc, yc])

    shortest_dist = np.inf
    for point in intersection_points:
        dist = np.linalg.norm(point-robot_pos)
        if (dist < shortest_dist):
            shortest_dist = dist

    r = np.linalg.norm(obstacle_pos-robot_pos) - shortest_dist 

    #print(r)

    return r   


