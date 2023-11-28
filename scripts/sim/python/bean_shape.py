from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def bean_shape(x_pos, y_pos, a=1, b=1.1):
    
    x,y = symbols('x y')
    # Shape parameters
    x_ci1 = x_pos
    y_ci1 = y_pos

    beta_star = ((x-x_ci1-a)**2 + (y-y_ci1)**2) * ((x-x_ci1+a)**2 + (y-y_ci1)**2) - b**4
    print(beta_star)
    return beta_star
