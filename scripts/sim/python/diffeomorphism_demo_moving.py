import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, atan2


def calculate_theta(x, xi):
    theta = atan2(x[1]-xi[1], x[0]-xi[0])
    return theta

""" 
def diffeomorphismF(M, x, xi, x_g, rho_i, qi, q_g):
    lam = 100
    #a = 1
    #b = 1.1
    gamma = (np.linalg.norm(x-x_g))**2
    F_list = []

    beta_i = []
    beta_i.append(rho_i[0]**2 - ((x[0]-xi[0][0])**2 + (x[1]-xi[0][1])**2))
    for i in range(1,M):
        beta_i.append((x[0]-xi[i][0])**2 + (x[1]-xi[i][1])**2 - rho_i[i]**2)

    beta_dash_i = {}
    if len(beta_i) > 1:
        for i in range(0,M):
            p = []
            for j in range(0,M):
                if i != j:
                    p.append(beta_i[j])
            beta_dash_i[i] = np.prod(p)
    else:
        beta_dash_i[0] = 0.0
    sigma = 0.0
    for i in range(0,M):
        sigma_i = (gamma*beta_dash_i[i])/(gamma*beta_dash_i[i]+lam*beta_i[i])
        sigma += sigma_i
        theta = calculate_theta(x,xi[i])
        ri = rho_i[i]
        fi = (np.linalg.norm(x-xi)/ri)*np.array([cos(theta), sin(theta)])
        l = sigma_i*(rho_i[i]*fi+qi[i])
        F_list.append(l)
    sigma_g = 1 - sigma
    F_list.append(sigma_g * (x-x_g+q_g))
    F = sum(F_list)
    return F
 """

def diffeomorphismF(M, x, xi, x_g, rho_i, qi, q_g):

    r_rw = [5.0, 0.5]

    lam = 100
    a = 1
    b = 1.1
    gamma = (np.linalg.norm(x-x_g))**2
    F_list = []

    beta_i = []
    #beta_i.append(rho_i[0]**2 - np.linalg.norm(x-xi[0])**2)
    beta_i.append(rho_i[0]**2 - ((x[0]-xi[0][0])**2 + (x[1]-xi[0][1])**2))

    for i in range(1,M):
        #beta_i.append(((x[0]-xi[i][0]-a)**2 + (x[1]-xi[i][1])**2)*((x[0]-xi[i][0]+a)**2 + (x[1]-xi[i][1])**2) - b**4)
        #beta_i.append(np.linalg.norm(x-xi[i])**2 - rho_i[i]**2)
        beta_i.append(((x[0]-xi[i][0])**2 + (x[1]-xi[i][1])**2) - rho_i[i]**2)
    
    beta_dash_i = {}
    for i in range(0,M):
        p = []
        for j in range(0,M):
            if i != j:
                p.append(beta_i[j])
        beta_dash_i[i] = np.prod(p)

    sigma = 0.0
    for i in range(0,M):
        sigma_i = (gamma*beta_dash_i[i])/(gamma*beta_dash_i[i]+lam*beta_i[i])
        sigma += sigma_i
        theta = calculate_theta(x,xi[i])
        if i == 0:
            ri = r_rw[0]
        else:
            ri = r_rw[1]
            #ri = calculate_r(x, xi[i])
        fi = (np.linalg.norm(x-xi[i])/ri)*np.array([np.cos(theta), np.sin(theta)]).T

        # This is for the old paper diffeomorphism
        """    
        if i == 0:
            fi = rho_i[i] * ((1-beta_i[i]) / (np.linalg.norm(x-xi[i])))
        else:
            fi = rho_i[i] * ((1+beta_i[i]) / (np.linalg.norm(x-xi[i])))
        l = sigma_i*(fi * (x-xi[i]) + qi[i])
        
        """
        l = sigma_i*(rho_i[i]*fi+qi[i])
        F_list.append(l)
    sigma_g = 1 - sigma
    F_list.append(sigma_g * (x-x_g+q_g))
    F = sum(F_list)
    #print(sigma_g * (x-x_g+q_g))
    #print(F)
    return F

grid_size = 7
x_initial, y_initial = np.meshgrid(np.linspace(-2.5, 2.5, grid_size), np.linspace(-2.5, 2.5, grid_size))
x_initial = x_initial.flatten()
y_initial = y_initial.flatten()
xi = np.array([[0., 0.], [1., 1.], [-1., -1.]])
rho_i = np.array([5.0, 0.5, 0.5])

def update_positions(x_initial, y_initial, xi, rho_i):
    x_ = np.array([x_initial[0], y_initial[0]])
    qi = xi
    x_g = np.array([2., 2.])
    #M = len(xi)
    M = 2
    q_g = x_g
    x_updated = np.zeros((len(x_initial)))
    y_updated = np.zeros((len(y_initial)))
    for i in range(len(x_initial)):
        x_ = np.array([x_initial[i], y_initial[i]])
        F = diffeomorphismF(M, x_, xi, x_g, rho_i, qi, q_g)
        x_updated[i] = F[0]
        y_updated[i] = F[1]
    return x_updated, y_updated

x_updated, y_updated = update_positions(x_initial, y_initial, xi, rho_i)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(x_initial, y_initial, color='blue', label='Initial Positions')
ax1.set_title('Initial Positions')
ax1.set_xlim([-4, 4])
ax1.set_ylim([-4, 4])

scatter = ax2.scatter([], [], color='red', label='Updated Positions')
obstacle = plt.Circle((xi[1][0], xi[1][1]), rho_i[1], color='red', label='obstacle')
obstacle2 = plt.Circle((xi[2][0], xi[2][1]), rho_i[2], color='green', label='obstacle')
ax2.add_patch(obstacle)
#ax2.add_patch(obstacle2)
ax2.set_title('Updated Positions')
ax2.set_xlim([-4, 4])
ax2.set_ylim([-4, 4])

arrows = []
for i in range(grid_size**2):
    arrow = ax2.arrow(x_initial[i], y_initial[i], x_updated[i] - x_initial[i], y_updated[i] - y_initial[i],
                      color='black', shape='full', lw=0.1, length_includes_head=True, head_width=0.1)
    arrows.append(arrow)

def update_plot(xi, scatter, obstacle, arrows):
    xi[1][0] += np.random.randn() * 0.1
    xi[1][1] -= 0.05
    rho_i[1] += np.random.randn() * 0.1
    obstacle.set_radius(rho_i[1])
    x_updated, y_updated = update_positions(x_initial, y_initial, xi, rho_i)
    obstacle.set_center((xi[1][0], xi[1][1]))
    scatter.set_offsets(np.column_stack((x_updated, y_updated)))
    obstacle.set_center((xi[1][0], xi[1][1]))

    for i in range(grid_size**2):
        new_arrow = ax2.arrow(x_initial[i], y_initial[i], x_updated[i] - x_initial[i], y_updated[i] - y_initial[i],
                          color='black', shape='full', lw=0.1, length_includes_head=True, head_width=0.1)
        arrows[i].remove()
        arrows[i] = new_arrow
    plt.pause(0.1)

i = 0
while i < 100:
    update_plot(xi, scatter, obstacle, arrows)
    plt.pause(0.1)
    ax2.figure.canvas.draw()
    i += 1

#plt.tight_layout()
#plt.show()
