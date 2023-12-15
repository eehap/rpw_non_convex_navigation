import numpy as np
import matplotlib.pyplot as plt
from r_function import calculate_r

#def calculate_r():
 #   r = 1.0
  #  return r


def calculate_theta(x, xi):
    theta = np.arctan2(x[1]-xi[1], x[0]-xi[0])
    return theta

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
        l = sigma_i*(rho_i[i]*fi+qi[i])
        F_list.append(l)
    sigma_g = 1 - sigma
    F_list.append(sigma_g * (x-x_g+q_g))
    F = sum(F_list)
    #print(sigma_g * (x-x_g+q_g))
    #print(F)
    return F

def create_mesh_grid(field_x, field_y, grid_size=0.1):
        aa=0
        m = int( (field_x[1]+aa - field_x[0]-aa) // grid_size ) 
        n = int( (field_y[1]+aa - field_y[0]-aa) // grid_size ) 
        gx, gy = np.meshgrid(np.linspace(field_x[0]-aa, field_x[1]+aa, m), np.linspace(field_y[0]-aa, field_y[1]+aa, n))
        return gx.flatten(), gy.flatten()

def main():
    x = np.array([-2.0, 0.0])
    x_g = np.array([2.0, 0.0])
    xi = np.array([[0.0, 0.0], [2.0, 2.0]])
    M = 2
    qi = xi
    q_g = x_g

    ##-- Change thease to bend the ball world --##
    rho_i = np.array([5.0, 0.5])       # Ball world radius, obstacle radius

    field_x = np.array([-2.5, 2.5])
    field_y = np.array([-2.5, 2.5])


    gx, gy = create_mesh_grid(field_x, field_y)
    bx = np.zeros((len(gx)))
    by = np.zeros((len(gy)))

    phi = np.linspace(0, 2*np.pi, len(gx))
    rgb_cycle = (np.stack((np.cos(phi          ),
                        np.cos(phi+2*np.pi/3),
                        np.cos(phi-2*np.pi/3)
                        )).T
                + 1)*0.5
    
    plt.scatter(gx, gy, c=rgb_cycle)

    for i in range(0, len(gx)):
        px = gx[i]
        py = gy[i]

        p = np.array([px, py]) 
        p = diffeomorphismF(M, p, xi, x_g, rho_i, qi, q_g)
        bx[i] = p[0]
        by[i] = p[1]

    plt.figure(2)
    plt.scatter(bx, by, c=rgb_cycle)
    plt.show()

if __name__ == '__main__':
    main()