from sympy import *
import math
from star_shape import star_shape
#import numpy as np
from bean_shape import bean_shape

init_printing(use_unicode=True)


def calculate_r(robot_position, obstacle_position):
    x, y, xc, yc, r, m, b, h = symbols('x y xc yc r m b h')

    a = 1
    b = 1.1
    
    # we can assume the shape to be at (0,0) for simplicity
    x_ci1 = 0
    y_ci1 = 0
    theta = math.atan2(robot_position[1]-obstacle_position[1], robot_position[0]-obstacle_position[0])

    x_expr = r*cos(theta)
    y_expr = r*sin(theta)

    bean = ((x_expr-x_ci1-a)**2 + (y_expr-y_ci1)**2) * ((x_expr-x_ci1+a)**2 + (y_expr-y_ci1)**2) - b**4
    #bean = 0.5**2 - ((x_expr-obstacle_position[0])**2 + (y_expr-obstacle_position[1])**2)
    #bean = bean.subs({theta: theta_val})
    solutions = solve(bean,r)
    for sol in solutions:
        if sol.is_real and sol > 0:
            r = sol

    return r, theta


