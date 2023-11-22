from sympy import *
import math
from star_shape import star_shape
import numpy as np

init_printing(use_unicode=True)

x, y, xc, yc, r, m, b, h = symbols('x y xc yc r m b h')

x_robot_pos = -3
y_robot_pos = -3

xc = 0
yc = 0

# Calculate angle
x_dif = x_robot_pos - xc
y_dif = y_robot_pos - yc

m = y_dif/x_dif
b = y_robot_pos - (m*x_robot_pos)

x_robot = Matrix([[x], [y]])
x_obs_1 = Matrix([[xc], [yc]])

matrix = (x_robot-x_obs_1).T @ Matrix([[10,0],[0,-1]]) @ (x_robot-x_obs_1)

#h_function = Eq((x - xc)**4 + (y - yc)**4 - matrix[0,0],y) # This is h-function
#h_function = Eq((x - xc)**2 + (y - yc)**2, 1**2) # This is for sphere for testing purposes
star_shape_func =  solve(star_shape(0,0,4,1),y)
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
print("Intersection points:", intersection_points)

# Solve for closest intersection point and calculate r
robot_pos = np.array([x_robot_pos, y_robot_pos])
obstacle_pos = np.array([xc, yc])

shortest_dist = np.inf
for point in intersection_points:
    dist = np.linalg.norm(point-robot_pos)
    if (dist < shortest_dist):
        shortest_dist = dist

r = np.linalg.norm(obstacle_pos-robot_pos) - shortest_dist    
print(r)


