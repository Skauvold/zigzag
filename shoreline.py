import matplotlib.pyplot as plt
import numpy as np

xmin = 0.0
xmax = 500.0
spacing = 0.06 * (xmax - xmin)
rate = 0.1 * (xmax - xmin)

z0 = 0.12
dz = 0.07

xx = np.linspace(xmin, xmax, 200)


def bathy(x):
    return np.maximum(np.minimum(1.0, np.exp(-x/rate)), 0.00)


def dist(y):
    return -rate * np.log(y)


def foreshore_profile(x):
    return np.exp(-x/rate)


class Clinothem:
    def __init__(self, x_left, x_right, bathymetry, dx_shoreline, dz_shoreline):
        self.x_left_ = x_left
        self.x_break_ = x_left + dx_shoreline
        self.x_right_ = x_right
        self.bottom_ = bathymetry
        self.z_left_ = bathymetry[0]
        self.z_right_ = bathymetry[-1] + dz_shoreline
        self.z_break_ = bathymetry[0] + dz_shoreline
        
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

    def get_transition_fill_xy(self, wh, n):
        z_wb = self.z_break_ - wh
        xs, zs = self.get_fill_xy(n)
        xf = xs[np.nonzero(zs <= z_wb)]
        zf = zs[np.nonzero(zs <= z_wb)]
        return xf, zf


def fill_rect_xy(x, y):
    return [x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]]

current_bathymetry = bathy(xx)
current_break_x = 0.0
n_obj = 10
sea_level_rise = np.random.uniform(0.0, 0.05, n_obj)
progradation = np.random.uniform(17.0, 22.0, n_obj)

plt.plot(xx, current_bathymetry, "k-")

x_below, y_below = fill_rect_xy([xmin, xmax], [-0.25, sum(sea_level_rise)])
plt.fill(x_below, y_below, "darkgray")
x_landward, y_landward = fill_rect_xy([xmin, xmin + sum(progradation)], [-0.25, 1.0 + sum(sea_level_rise)])
plt.fill(x_landward, y_landward, "darkgray")

x_lagoon, y_lagoon = fill_rect_xy([xmin, xmin + sum(progradation)], [1.0, 1.0 + sum(sea_level_rise)])
plt.fill(x_lagoon, y_lagoon, "seagreen")

x_sea, y_sea = fill_rect_xy([xmin + sum(progradation), xmax], [sum(sea_level_rise), 1.0 + sum(sea_level_rise)])
plt.fill(x_sea, y_sea, "skyblue", alpha=0.3)

for i in range(n_obj):
    c = Clinothem(current_break_x, xmax, current_bathymetry, progradation[i], sea_level_rise[i])
    current_break_x += progradation[i]

    xs, ys = c.get_fill_xy(100)
    plt.fill(xs, ys, "yellow")

    xt, yt = c.get_transition_fill_xy(0.95, 100)
    plt.fill(xt, yt, "tan")

    cx = np.linspace(current_break_x, xmax, 100)
    current_bathymetry = np.array([c.get_top(x) for x in cx])

    plt.plot(cx, current_bathymetry, "k--", linewidth=0.5)

plt.show()
