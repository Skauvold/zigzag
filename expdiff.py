import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(0.0, 1000.0, 100)


def f(x, a, b, zl, zr, lam):
    return np.array([(zr + (zl - zr) * np.exp(-(x-a)/lam)) if a <= x <= b else zl if x < a else zr for x in xx])


def taper(xx, s, t):
    return np.array([1.0 if x < s else 0.0 if x > t else 1.0 - ((x - s)/(t - s))**2 for x in xx])


plt.plot(xx, f(xx, 100.0, 600.0, 30.0, 5.0, 50.0))
plt.plot(xx, f(xx, 150.0, 500.0, 32.0, 5.0, 50.0))
#plt.plot(xx, f(xx, 150.0, 500.0, 32.0, 5.0, 50.0) - f(xx, 100.0, 600.0, 30.0, 5.0, 50.0))


delta_b = 32.0 - f([150.0], 100.0, 600.0, 30.0, 5.0, 50.0)
plt.plot(xx, np.minimum(32.0, f(xx, 100.0, 600.0, 30.0, 5.0, 50.0) + delta_b * np.exp(-(xx-150.0)/50.0)), "k--")
plt.plot(xx, np.minimum(32.0, f(xx, 100.0, 600.0, 30.0, 5.0, 50.0) + delta_b * np.exp(-(xx-150.0)/50.0) * taper(xx, 100.0, 500.0)), "k-.")

plt.show()
