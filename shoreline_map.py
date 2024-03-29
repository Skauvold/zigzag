import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift


def cov(x, sigma, crange, pow):
    return sigma**2 * np.exp(-np.power(np.abs(x)/(3*crange), pow))


def grf(xx, covfun):
    cc = covfun(np.hstack((np.flipud(xx), xx)))
    zz = np.random.normal(0.0, 1.0, cc.size)
    #zz = np.hstack((zz, np.flipud(zz)))
    u = ifft(fft(cc) * zz)
    return np.real(u[0:xx.size])


xx = np.linspace(0.0, 500.0, 100)

def shoreline_trend(x):
    return 100.0 * np.exp(-((x - 300.0)/10000.0)**2)

def increment_trend(x):
    return 8.0 * np.exp(-((x - 250.0)/200.0)**2)

plt.figure()
initial_shoreline = shoreline_trend(xx) + grf(xx, lambda x: cov(x, 10.0, 100.0, 2.0))
plt.plot(xx, initial_shoreline, "k-")
current_shoreline = initial_shoreline

for i in range(20):
    shoreline_increment = increment_trend(xx) * np.exp(grf(xx, lambda x: cov(x, 1.2, 12.0, 2.0)))
    current_shoreline += shoreline_increment
    plt.plot(xx, current_shoreline, "r-")

plt.show()
