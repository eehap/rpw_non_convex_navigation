from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def beta(x,y, xc, yc):
    a = 1
    b = 1.1
    function = ((x-xc-a)**2 + (y-yc)**2)*((x-xc+a)**2 + (y-yc)**2) - b**4
    return function


def beta0(x, y, r):
    return x**2 + y**2 - r**2
    

def main():
    x_values = np.linspace(-5, 5, 100)
    y_values = np.linspace(-5, 5, 100)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    xc1 = 0.
    yc1 = 3.
    xc2 = 0.
    yc2 = -3.
    r = 5.
    beta_values0 = beta0(x_mesh, y_mesh, r)
    beta_values = beta(x_mesh, y_mesh, xc1, yc1)
    beta_values2 = beta(x_mesh, y_mesh, xc2, yc2)

    plt.contour(x_mesh, y_mesh, beta_values, levels=[0], colors='r')
    plt.contour(x_mesh, y_mesh, beta_values2, levels=[0], colors='b')
    plt.contour(x_mesh, y_mesh, beta_values0, levels=[0], colors='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Non-convex Object')
    plt.grid(True)
    plt.show()
main()
