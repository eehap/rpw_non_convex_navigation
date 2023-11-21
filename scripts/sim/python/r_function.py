from sympy import *
import math

init_printing(use_unicode=True)

x, y, xc, yc, r, m, b = symbols('x y xc yc r m b')

x_robot = 3
y_robot = 3

xc = 0
yc = 0

# TODO
# need to solve b for our case
b = 0

# Calculate angle
x_dif = x_robot - xc
y_dif = y_robot - yc

m = y_dif/x_dif

x_robot = Matrix([[x], [y]])
x_obs_1 = Matrix([[xc], [yc]])

matrix = (x_robot-x_obs_1).T @ Matrix([[10,0],[0,-1]]) @ (x_robot-x_obs_1)

#h_function = Eq((x - xc)**4 + (y - yc)**4 - matrix[0,0],y) # This is h-function
h_function = Eq((x - xc)**2 + (y - yc)**2, 1**2) # This is for sphere for testing purposes

line_equation = Eq(y, m*x + b)

# Substitute y from the line equation into the h function
intersection_equation = h_function.subs(y, m*x + b)
# Solve for x to find the x-coordinates of intersection points
x_values = solve(intersection_equation, x)
print('x_values: ' ,x_values)

# Substitute x values back into the line equation to find y-coordinates
intersection_points = [(x_val, line_equation.subs(x, x_val).rhs) for x_val in x_values]

print("Intersection points:", intersection_points)


import numpy as np
import matplotlib.pyplot as plt

# Ellipse parameters
h = 0   # x-coordinate of the center
k = 0   # y-coordinate of the center
a1 = 4   # semi-major axis length
b1 = 1.5   # semi-minor axis length

angle = np.pi/2

rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

# Create a 2x2 matrix A
A1 = np.array([[1/a1**2, 0], [0, 1/b1**2]])
A2 = rotation_matrix @ A1

# Generate points on the ellipse
theta = np.linspace(0, 2*np.pi, 100)
ellipse_points = np.vstack((a1 * np.cos(theta), b1 * np.sin(theta)))

# Apply the matrix A to the ellipse points
transformed_ellipse = A1 @ ellipse_points
transformed_ellipse2 = A2 @ ellipse_points

# Plot the original ellipse
plt.figure(figsize=(8, 8))
#plt.plot(ellipse_points[0] + h, ellipse_points[1] + k, label='Original Ellipse')

# Plot the transformed ellipse using matrix A
plt.plot(transformed_ellipse[0] + h, transformed_ellipse[1] + k, label='Transformed Ellipse (Using Matrix A1)')
plt.plot(transformed_ellipse2[0] + h, transformed_ellipse2[1] + k, label='Transformed Ellipse (Using Matrix A2)')


# Set plot attributes
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.title('Ellipse and Matrix Transformation')
plt.legend()
plt.show()

