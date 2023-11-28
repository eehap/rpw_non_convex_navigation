from sympy import *
import numpy as np
from star_shape import star_shape

init_printing(use_unicode=True)

x, y, r_i, sigma, x_r, y_r, x_i_x, x_i_y = symbols('x y r_i sigma x_r y_r x_i_x x_i_y')

lambda_value = 100
kappa = 1

x_robot = [x, y]
#pitäisikö x_robot olla x_robot = [x_r, y_r]

#pitääkö kaikille esteille olla omat x_i_ ja x_i_y?
# esim  [x_1_x, x_1_y], [x_2_x, x_2_y] 
x_i = [x_i_x, x_i_y]

f_i = (sqrt((x_robot[0] - x_i[0])**2 + (x_robot[1] - x_i[1])**2) / r_i) * Matrix([[cos(sigma)], [sin(sigma)]])

#print(f'f_i: {shape(f_i)}')

x_g_x, x_g_y = symbols('x_g_x x_g_y')
x_g = [x_g_x, x_g_y]
gamma_g = sqrt((x_robot[0] - x_g[0])**2 + (x_robot[1] - x_g[1])**2)**2

#print(f'gamma_g: {gamma_g}')


beta_i = symbols('beta_i')

# give coords of shapes each time or store the betas in somewhere.
# Do we need world as a beta function? beta_i when i==0?
beta_i,_ = star_shape(0,0)
beta_j,_ = star_shape(2,3)
betas = [beta_i,beta_j]
#print(f'beta_i: {beta_i}')

#Pallomaailman esteiden siajinti ja säde?
rho_i = np.array([[0, 0], [-1, 1]])
q_i = np.array([1.37, 2.0])

##-- F(x) --##
# First part of equation
sigma_i_sum = None
for i in range(len(betas)):
    beta_i_bar = 0
    for j in range(len(betas)):
        if j != i:
            if beta_i_bar == 0:
                beta_i_bar = betas[j]
            else:
                beta_i_bar *= betas[j]

    rho_i_f = Matrix([[rho_i[i,0]*f_i[0]],[rho_i[i,1]*f_i[1]]])
    rho_i_f_q = Matrix([[rho_i_f[0]+q_i[i]],[rho_i_f[1]+q_i[i]]])

    if sigma_i_sum == None:
        sigma_i_sum = Matrix([[
                        ((gamma_g*beta_i_bar) / (gamma_g*beta_i_bar + lambda_value*betas[i])) * (rho_i_f_q[0]),
                        ((gamma_g*beta_i_bar) / (gamma_g*beta_i_bar + lambda_value*betas[i])) * (rho_i_f_q[1])]])
    else:
        sigma_i_sum += Matrix([[
                        ((gamma_g*beta_i_bar) / (gamma_g*beta_i_bar + lambda_value*betas[i])) * (rho_i_f_q[0]),
                        ((gamma_g*beta_i_bar) / (gamma_g*beta_i_bar + lambda_value*betas[i])) * (rho_i_f_q[1])]])
    
    #print(f'sigma_i_sum: {sigma_i_sum}')
#print(f'sigma_i_sum: {sigma_i_sum}')


# Second part of equation

q_g_x, q_g_y = symbols('q_g_x q_g_y')

q_g = [q_g_x, q_g_y]

sigma_g_sum = 0
for i in range(len(betas)):
    beta_i_bar = 0
    for j in range(len(betas)):
        if j != i:
            if beta_i_bar == 0:
                beta_i_bar = betas[j]
            else:
                beta_i_bar *= betas[j]

    sigma_g_sum += (gamma_g*beta_i_bar) / (gamma_g*beta_i_bar + lambda_value*betas[i])
    print(f'sigma_g_sum: {sigma_g_sum}')

sigma_g_lambda = 1 - sigma_g_sum
x_xg_qg = Matrix([x_robot[0]-x_g[0]+q_g[0], x_robot[1]-x_g[1]+q_g[1]])
sigma_g_lambda_x_xg_qg = Matrix([[sigma_g_lambda*x_xg_qg[0],sigma_g_lambda*x_xg_qg[1]]])
#print(f'Sigma_g_lambda: {sigma_g_lambda_x_xg_qg}')

final_F = sigma_i_sum + sigma_g_lambda_x_xg_qg
#print(f'Final F: {shape(final_F)}')
print(f'Final F first: {final_F[0]}')
#print(f'Final F second: {final_F[1]}')

#print(f'Final F first subs: {final_F.subs({x_i_x:5, x_i_y:3, r_i:1, sigma:0.7, x_g_x:0,x_g_y:0, y_r:1,x_r:1,q_g_x:3})}')
#print(f'Final F first subs: {diff(final_F[0], x)}')
jacob_1_1 = diff(final_F[0], x)
jacob_1_2 = diff(final_F[0], y)
jacob_2_1 = diff(final_F[1], x)
jacob_2_2 = diff(final_F[1], y)

Jacobian = Matrix([[jacob_1_1, jacob_1_2], [jacob_2_1, jacob_2_2]])
print(f'Jacob : {Jacobian}')
jacob_subs=Jacobian.subs({x_i_x:5, x_i_y:3, r_i:1, sigma:0.7, x_g_x:0,x_g_y:0, y_r:1,x_r:1,q_g_x:3,q_g_y:2})
#print(f'Final F first subs: {jacob_subs}')


print(f'Jacob inverse: {jacob_subs.inv()}')







