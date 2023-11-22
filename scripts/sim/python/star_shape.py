from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def star_shape(x_pos,y_pos,semi_major_axis, semi_minor_axis):
    x,y,h,k,r1,r2 = symbols('x y h k r1 r2')
    # Ellipse parameters
    h = x_pos   # x-coordinate of the center
    k = y_pos   # y-coordinate of the center
    a1 = semi_major_axis   # semi-major axis length
    b1 = semi_minor_axis   # semi-minor axis length
    r1 = 1
    r2 = 1

    angle = np.pi/2

    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Create a 2x2 matrix A
    A1 = np.array([[1/a1**2, 0], [0, 1/b1**2]])
    A2 = rotation_matrix @ A1 @ rotation_matrix.T

    x_ellips = Matrix([[x],[y]])
    x_ellips_i = Matrix([[h],[k]])
    ellips1 = (x_ellips-x_ellips_i).T * A1 * (x_ellips-x_ellips_i)
    ellips2 = (x_ellips-x_ellips_i).T * A2 * (x_ellips-x_ellips_i)

    beta_star = (ellips1[0,0] - r1**2) * (ellips2[0,0]-r2**2)
    #print(beta_star)


    #---FOR PLOTTING---#
    # Convert the expression to a function
    beta_star_y = solve(beta_star,y)
    beta_star_func = lambdify([x], beta_star_y, "numpy")
   
    # Given x-values
    x_values = np.linspace(-4,4,500)

    # Calculate corresponding y-values
    points = []
    for x_val in x_values:
        y_val = beta_star_func(x_val)
        for y in y_val:
            points.append([x_val, y])
        #print(f"For x = {x_val}, y = {y_val}")
    points = np.array(points)

    plt.scatter(points[:,0], points[:,1])

    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.title('Star shape')
    plt.legend()
    plt.show()
    #---FOR PLOTTING---#    

    return beta_star