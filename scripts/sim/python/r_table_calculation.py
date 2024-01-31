from math import atan2, pi
import numpy as np
import sympy as sp
import json

x, y, r, theta = sp.symbols('x y r theta')
xc =  0.0
yc = 3.0
a = 1
b = 1.1
equation = ((x - xc - a)**2 + (y - yc)**2)*((x - xc + a)**2 + (y - yc)**2) - 1.1**4

#90: r = 0.458
#0: r= 1.485

xlin = np.linspace(-1.4862, 1.4862, 300)
ylin = np.linspace(0.0, 0.0, 300)

for i in range(len(xlin)):
    x_ = xlin[i]
    e = equation.subs(x, x_)
    sol = sp.solve(e, y)[1]
    ylin[i] = sol

thetas = np.linspace(0.0, 0.0, 300)
radius = np.linspace(0.0, 0.0, 300)
table = {}
for i in range(len(xlin)):
    ri = np.linalg.norm(np.array([xlin[i], ylin[i]]) - np.array([xc, yc]))
    radius[i] = ri
    th = atan2(ylin[i]-yc, xlin[i]-xc)
    th = th*180/pi
    th = round(th)
    thetas[i] = th
    table[th] = ri
    
r_table = dict(sorted(table.items()))
file_path = 'r_table_new.json'

with open(file_path, 'w') as file:
    json.dump(r_table, file)

print(file_path)
