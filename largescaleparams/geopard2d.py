# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:46:01 2022

Define 2D Geopard surrogate model

@author: jas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, minimize, optimize, differential_evolution

#%% X-axis discretization
x_min = 0.0
x_max = 100.0
n_x = 100
pts_x = np.linspace(x_min, x_max, n_x)

# Base surface
def basesurface(x, params):
    x_break = params[0]
    z_proximal = params[1]
    z_distal = params[2]
    scale = params[3]
    return z_proximal * (x < x_break) + (z_proximal + (1.0 - np.exp(-(x - x_break)/scale)) * (z_distal - z_proximal)) * (x >= x_break)

# Top surface
def topsurface(object_params, n):
    x_left = object_params[2]
    x_right = object_params[3]
    y_left = object_params[6]
    y_right = object_params[7]
    scale = object_params[8]
    xx = np.linspace(x_left, x_right, n)
    yy = y_left + (1.0 - np.exp(-(xx - x_left) / scale)) * (y_right - y_left)
    return xx, yy

def topsurfacepoint(x, x_left, x_right, y_left, y_right, scale):
    return y_left + (1.0 - np.exp(-(x - x_left) / scale)) * (y_right - y_left)

def getfacies(x, y, base_surface_params, object_params):    
    x_left_base = object_params[0]
    x_right_base = object_params[1]
    x_left_top = object_params[2]
    x_right_top = object_params[3]
    y_left_base = object_params[4]
    y_right_base = object_params[5]
    y_left_top = object_params[6]
    y_right_top = object_params[7]
    object_scale = object_params[8]
    
    if x < x_left_base:
        return 0
    elif x < x_left_top:
        y_base = basesurface(x, base_surface_params)
        y_top = y_left_base + (y_left_top - y_left_base) * (x - x_left_base) / (x_left_top - x_left_base)
        if y_base < y < y_top:
            return 1
        else:
            return 0
    elif x < x_right_base:
        y_base = basesurface(x, base_surface_params)
        y_top = topsurfacepoint(x, x_left_top, x_right_top, y_left_top, y_right_top, object_scale)
        if y_base < y < y_top:
            return 1
        else:
            return 0
    elif x < x_right_top:
        y_base = y_right_base + (y_right_top - y_right_base) * (x - x_right_base) / (x_right_top - x_right_base)
        y_top = topsurfacepoint(x, x_left_top, x_right_top, y_left_top, y_right_top, object_scale)
        if y_base < y < y_top:
            return 1
        else:
            return 0
    else:
        return 0

def obsmismatch(pathx, pathy, facies, base_par, obj_par):
    accumulated_mismatch = 0.0
    for (x, y, f) in zip(pathx, pathy, facies):
        if getfacies(x, y, base_par, obj_par) != f:
            accumulated_mismatch += 1.0
    return accumulated_mismatch

def datamismatch(data, base_par, obj_par, ds):
    mm = 0.0
    for obs in data:
        x_obs_reg, y_obs_reg, f_obs_reg = regularizeobs(obs[0], obs[1], obs[2], ds)
        mm += ds * obsmismatch(x_obs_reg, y_obs_reg, f_obs_reg, base_par, obj_par)
    return mm

def regularizeobs(pathx, pathy, facies, ds):
    npath = len(pathx)
    xs = []
    ys = []
    fs = []
    for i in range(npath - 1):
        x0 = pathx[i]
        x1 = pathx[i + 1]
        y0 = pathy[i]
        y1 = pathy[i + 1]
        segment_length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        n_segment = int(np.round(segment_length / ds))
        xs += [x for x in np.linspace(x0, x1, n_segment)]
        ys += [y for y in np.linspace(y0, y1, n_segment)]
        fs += [facies for j in range(n_segment)]
    return xs, ys, fs
        
def makeobject(theta, baseparams):
    x_left_base = theta[0]
    object_length = theta[1]
    prog = theta[2]
    agg = theta[3]
    x_right_base = x_left_base + object_length
    y_left_base = basesurface(x_left_base, baseparams)
    y_right_base = basesurface(x_right_base, baseparams)
    y_left_top = y_left_base + agg
    y_right_top = y_right_base + agg
    x_left_top = x_left_base + prog
    x_right_top = x_right_base + prog
    object_params = [x_left_base, x_right_base, x_left_top, x_right_top, y_left_base, y_right_base, y_left_top, y_right_top, 10.0]
    return object_params

def obscolor(facies):
    if facies == 0:
        return "red"
    elif facies == 1:
        return "green"
    else:
        return "gray"

#%% Create base surface and draw object

base_surface_params = [20.0, 10.0, -30.0, 10.0]

x_left_base = np.random.uniform(20.0, 40.0, 1)
object_length = np.random.uniform(30.0, 50.0, 1)
prog = np.random.uniform(5.0, 15.0, 1)
agg = np.random.uniform(1.0, 3.0, 1)

object_params = makeobject([x_left_base, object_length, prog, agg], base_surface_params)

# Visualize object
plt.plot(pts_x, basesurface(pts_x, base_surface_params), linewidth=2, color="green")
x_top, y_top = topsurface(object_params, 100)
plt.plot(x_top, y_top, linewidth=2, color="red")
plt.show()

y_min = -30.0
y_max = 10.0
n_y = 80        
pts_y = np.linspace(y_min, y_max, n_y)
facies = np.empty(shape=(n_x, n_y))

for i, x in enumerate(pts_x):
    for j, y in enumerate(pts_y):
        facies[i, j] = getfacies(x, y, base_surface_params, object_params)

plt.imshow(facies.transpose(), extent=(x_min, x_max, y_min, y_max), origin='lower')

x_obs_0 = [40.0, 40.0]
y_obs_0 = [0, -15.0]
f_obs_0 = 0

x_obs_1 = [40.0, 40.0]
y_obs_1 = [-15.0, basesurface(40.0, base_surface_params)]
f_obs_1 = 1

x_obs_2 = [60.0, 60.0]
y_obs_2 = [-10.0, basesurface(60.0, base_surface_params)]
f_obs_2 = 0

ds = 0.05

observations = [(x_obs_0, y_obs_0, f_obs_0), (x_obs_1, y_obs_1, f_obs_1), (x_obs_2, y_obs_2, f_obs_2)]

for obs in observations:
    x_obs, y_obs, f_obs = regularizeobs(obs[0], obs[1], obs[2], ds)
    plt.plot(x_obs, y_obs, "-", linewidth=4, color=obscolor(f_obs[0]))
    
plt.title(datamismatch(observations, base_surface_params, object_params, ds))

plt.show()

#%% Parameter profile

mu_theta = [
    30.0, # left base (distal edge)
    40.0, # length
    10.0, # progradation
    2.0   # aggradation
]

# Vary left base only
leftbase_range = np.linspace(0.5 * mu_theta[0], 2.0 * mu_theta[0], 40)
mismatch_values = np.empty_like(leftbase_range)
for i, theta_0 in enumerate(leftbase_range):
    theta_i = mu_theta.copy()
    theta_i[0] = theta_0
    object_params_i = makeobject(theta_i, base_surface_params)
    mismatch_i = datamismatch(observations, base_surface_params, object_params_i, ds)
    mismatch_values[i] = mismatch_i

plt.plot(leftbase_range, mismatch_values, "rs-")
plt.xlabel("theta_0 (left base x)")
plt.ylabel("obs. mismatch")
plt.title("mismatch profile")


#%% Vary left base and length
leftbase_range = np.linspace(0.1 * mu_theta[0], 2 * mu_theta[0], 40)
length_range = np.linspace(0.1 * mu_theta[1], 2 * mu_theta[1], 40)
mismatch_values_ll = np.empty(shape=(40, 40), dtype='float')
for i, leftbase_i in enumerate(leftbase_range):
    for j, length_j in enumerate(length_range):
        theta_ij = mu_theta.copy()
        theta_ij[0] = leftbase_i
        theta_ij[1] = length_j
        object_params_ij = makeobject(theta_ij, base_surface_params)
        mismatch_ij = datamismatch(observations, base_surface_params, object_params_ij, ds)
        mismatch_values_ll[i, j] = mismatch_ij

plt.contourf(leftbase_range, length_range, mismatch_values_ll, origin="lower")
plt.xlabel("left endpoint position")
plt.ylabel("length")
cb_handle = plt.colorbar()
plt.title("mismatch")
plt.show()


#%% Vary progradation and aggradation
progradation_range = np.linspace(0.1 * mu_theta[2], 2 * mu_theta[2], 40)
aggradation_range = np.linspace(0.1 * mu_theta[3], 2 * mu_theta[3], 40)
mismatch_values_pa = np.empty(shape=(40, 40), dtype='float')
for i, progradation_i in enumerate(progradation_range):
    for j, aggradation_j in enumerate(aggradation_range):
        theta_ij = mu_theta.copy()
        theta_ij[2] = progradation_i
        theta_ij[3] = aggradation_j
        object_params_ij = makeobject(theta_ij, base_surface_params)
        mismatch_ij = datamismatch(observations, base_surface_params, object_params_ij, ds)
        mismatch_values_pa[i, j] = mismatch_ij

plt.contourf(progradation_range, aggradation_range, mismatch_values_pa, origin="lower")
plt.xlabel("progradation")
plt.ylabel("aggradation")
cb_handle = plt.colorbar()
plt.title("mismatch")
plt.show()


#%% Optimize

bounds = Bounds(lb = [0.0, 0.0, 0.0, 0.0], ub = [2.0 * theta for theta in mu_theta])
# res = minimize(lambda theta: datamismatch(observations, base_surface_params, makeobject(theta, base_surface_params), ds),
#                [0.6 * theta for theta in mu_theta],
#                method='Nelder-Mead',
#                bounds=bounds)

res = differential_evolution(lambda theta: datamismatch(observations, base_surface_params, makeobject(theta, base_surface_params), ds),
                  bounds=((0.0, 1.6 * mu_theta[0]), (0.0, 1.6 * mu_theta[1]), (0.0, 1.6 * mu_theta[2]), (0.0, 1.6 * mu_theta[3])),
                  init='random')

print(res.x)
print(mu_theta)

# Show best-fitting object
for i, x in enumerate(pts_x):
    for j, y in enumerate(pts_y):
        facies[i, j] = getfacies(x, y, base_surface_params, makeobject(res.x, base_surface_params))

plt.imshow(facies.transpose(), extent=(x_min, x_max, y_min, y_max), origin='lower')

plt.plot(pts_x, basesurface(pts_x, base_surface_params), linewidth=2, color="green")

x_top, y_top = topsurface(makeobject(res.x, base_surface_params), 100)
plt.plot(x_top, y_top, linewidth=2, color="red")
for obs in observations:
    x_obs, y_obs, f_obs = regularizeobs(obs[0], obs[1], obs[2], ds)
    plt.plot(x_obs, y_obs, "-", linewidth=4, color=obscolor(f_obs[0]))
    
plt.title(datamismatch(observations, base_surface_params, makeobject(res.x, base_surface_params), ds))

plt.show()