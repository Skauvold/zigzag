import matplotlib.pyplot as plt
import numpy as np

xmin = 0.0
xmax = 200.0
spacing = 0.06 * (xmax - xmin)
rate = 0.1 * (xmax - xmin)

z0 = 0.12
dz = 0.07

xx = np.linspace(xmin, xmax, 200)

bathy = lambda x: np.maximum(np.minimum(1.0, np.exp(-x/rate)), 0.00)
dist = lambda y: -rate * np.log(y)

n_clinoforms = 8
i_slr = 2

def foreshore_profile(x):
    return np.exp(-x/rate)

class Clinothem:
    def __init__(self, x_left, x_right, bathymetry, dx_shoreline, dz_shoreline):
        self.x_left_ = x_left
        self.x_break_ = x_left + dx_shoreline
        self.x_right_ = x_right
        self.bottom_ = bathymetry
        self.z_left_ = bathymetry[0]
        self.z_right_ = bathymetry[-1]
        self.z_break_ = self.z_left_ + dz_shoreline
        
    def get_top(self, x):
        if self.x_left_ <= x <= self.x_break_:
            return self.z_left_ + (self.z_break_ - self.z_left_) * (x - self.x_left_) / (self.x_break_ - self.x_left_)
        elif self.x_break_ <= x <= self.x_right_:
            return self.z_right_ + (self.z_break_ - self.z_right_) * foreshore_profile(x - self.x_break_)
    
    def get_fill_xy(self, n):
        xb = np.linspace(self.x_left_, self.x_right_, self.bottom_.size)
        xt = np.linspace(self.x_right_, self.x_break_, n)
        xf = np.hstack((xb, xt))
        zt = np.array([self.get_top(x) for x in xt])
        zf = np.hstack((self.bottom_, zt))
        return xf, zf


initial_bathymetry = bathy(xx)

plt.plot(xx, initial_bathymetry, "k-")

c1 = Clinothem(0.0, xmax, bathy(xx), 30.0, 0.1)
x1, y1 = c1.get_fill_xy(100)
plt.fill(x1, y1, "orange")

c1x = np.linspace(30.0, xmax, 100)
c1top = np.array([c1.get_top(x) for x in c1x])

plt.plot(c1x, c1top, "k--")
c2 = Clinothem(30.0, xmax, c1top, 20.0, 0.01)
x2, y2 = c2.get_fill_xy(100)
plt.fill(x2, y2, "orange")

plt.show()
