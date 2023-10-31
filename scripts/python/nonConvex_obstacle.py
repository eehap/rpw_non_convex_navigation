import numpy as np
import matplotlib.pyplot as plt

def plot_nonConvex_shape(a=0.6, b=2, num_points=1000):
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # Using the equation to get the radii
    r = (1 + a * np.cos(b * theta)) # * 0.5   #(scale)
    
    # Convert polar coordinates to Cartesian
    x = r * np.cos(theta) # + 1   #(offset)
    y = r * np.sin(theta) # + 2   #(offset)

    # Add rotation
    rot = np.radians(0) # syötä kääntö asteina
    x_rot = x * np.cos(rot) - y * np.sin(rot)
    y_rot = x * np.sin(rot) + y * np.cos(rot)
    
    plt.figure(figsize=(6, 6))
    plt.plot(x_rot, y_rot, 'b') 
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Draw the shape
plot_nonConvex_shape()



