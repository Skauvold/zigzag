import matplotlib.pyplot as plt
import numpy as np

xmin = 0.0
xmax = 100.0
spacing = 0.06 * (xmax - xmin)
rate = 0.2 * (xmax - xmin)

z0 = 0.1
dz = 0.04

xx = np.linspace(xmin, xmax, 200)

bathy = lambda x: np.maximum(np.minimum(1.0, np.exp(-x/rate)), 0.05)
dist = lambda y: -rate * np.log(y)

n_clinoforms = 8
i_slr = 2

plt.plot(xx, bathy(xx), "k-")

for i in range(n_clinoforms):
    xv = np.hstack((xx, np.flipud(xx)))
    yv = np.hstack((bathy(xx - i * spacing), np.flipud(bathy(xx - (i + 1) * spacing))))
    plt.fill(xv, yv, "yellow")
    plt.plot(xx, bathy(xx - (i + 1) * spacing), "k-")

    #zc = z0 + dz * (i > i_slr)
    zc = z0 + dz * i
    xl = dist(zc) + i * spacing
    xr = xl + spacing

    plt.plot([xl, xr], [zc, zc], "k-")

    xb = np.linspace(xl, xmax, 200)
    xt = np.linspace(xr, xmax, 200)
    xu = np.hstack((xb, np.flipud(xt)))
    yu = np.hstack((bathy(xb - i * spacing), np.flipud(bathy(xt - (i + 1) * spacing))))
    plt.fill(xu, yu, "lightgray")

plt.show()