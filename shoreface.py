import matplotlib.pyplot as plt
import numpy as np

# Set up grid
xmin = 0.0
xmax = 500.0
xx = np.linspace(xmin, xmax, 200)
rate = 0.1 * (xmax - xmin)

def bathy(x):
    """ Shape of initial shoreface/seafloor """
    return np.maximum(np.minimum(1.0, np.exp(-x/rate)), 0.00)

def foreshore_profile(x):
    """ Clinoform shape """
    return np.exp(-x/rate)


class Bedset:
    def __init__(self, x_left, x_right, bathymetry, dx_shoreline, dz_shoreline):
        self.x_left_ = x_left
        self.x_break_ = x_left + dx_shoreline
        self.x_right_ = x_right
        self.bottom_ = bathymetry
        self.z_left_ = bathymetry[0]
        self.z_right_ = bathymetry[-1] + dz_shoreline
        self.z_break_ = bathymetry[0] + dz_shoreline
        
    def get_top(self, x):
        """ Evaluate top surface of bedset"""
        if self.x_left_ <= x <= self.x_break_:
            return self.z_left_ + (self.z_break_ - self.z_left_) * (x - self.x_left_) / (self.x_break_ - self.x_left_)
        elif self.x_break_ <= x <= self.x_right_:
            return self.z_right_ + (self.z_break_ - self.z_right_) * foreshore_profile(x - self.x_break_)
    
    def get_fill_xy(self, n):
        """ Get coordinates for coloring in whole bedset """
        xb = np.linspace(self.x_left_, self.x_right_, self.bottom_.size)
        xt = np.linspace(self.x_right_, self.x_break_, n)
        xf = np.hstack((xb, xt))
        zt = np.array([self.get_top(x) for x in xt])
        zf = np.hstack((self.bottom_, zt))
        return xf, zf

    def get_transition_fill_xy(self, wh, n):
        """ Get coordinates for coloring in bottom part of bedset """
        z_wb = self.z_break_ - wh
        xs, zs = self.get_fill_xy(n)
        xf = xs[np.nonzero(zs <= z_wb)]
        zf = zs[np.nonzero(zs <= z_wb)]
        return xf, zf


def fill_rect_xy(x, y):
    """ Convenience function for plotting filled rectangles"""
    return [x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]]


# Number of bedset objects to deposit
n_obj = 12

# Generate vertical and horizontal increments of shoreline trajectory
sea_level_rise = np.random.uniform(0.0, 0.07, n_obj)
progradation = np.random.uniform(15.0, 25.0, n_obj)

# Loop through objects, plotting intermediate states
for n_obj_plot in range(n_obj + 1):
    # Clear figure axis
    plt.cla()

    # Draw initial bathymetry
    current_bathymetry = bathy(xx)
    plt.plot(xx, current_bathymetry, "k-")

    current_break_x = 0.0
    current_sea_z = 1.0

    # Color in sky
    x_sky, y_sky = fill_rect_xy([xmin, xmax], [-0.25, 2.25])
    plt.fill(x_sky, y_sky, "lightskyblue", alpha=0.3)

    # Color in background (basement)
    x_bg = np.hstack((xx, xmax, xmin))
    y_bg = np.hstack((current_bathymetry, -0.25, -0.25))
    plt.fill(x_bg, y_bg, "darkgray")

    # Color in lagoon behind shoreline
    if n_obj_plot > 0:
        x_lagoon, y_lagoon = fill_rect_xy([xmin, xmin + sum(progradation[0:n_obj_plot])], [1.0, 1.0 + sum(sea_level_rise[0:n_obj_plot])])
        plt.fill(x_lagoon, y_lagoon, "seagreen")

        for i in range(n_obj_plot):
            c = Bedset(current_break_x, xmax, current_bathymetry, progradation[i], sea_level_rise[i])
            current_break_x += progradation[i]
            current_sea_z += sea_level_rise[i]

            # Color in bedset
            xs, ys = c.get_fill_xy(200)
            plt.fill(xs, ys, "yellow")

            # Color in bottom part of bedset (transition zone) with different color
            xt, yt = c.get_transition_fill_xy(0.95, 200)
            plt.fill(xt, yt, "tan")

            # Draw dashed "time line"
            cx = np.linspace(current_break_x, xmax, 200)
            current_bathymetry = np.array([c.get_top(x) for x in cx])
            plt.plot(cx, current_bathymetry, "k--", linewidth=0.5)

    # Color in sea
    x_sea = np.hstack((np.linspace(current_break_x, xmax, 200), xmax))
    y_sea = np.hstack((current_bathymetry, current_sea_z))
    plt.fill(x_sea, y_sea, "deepskyblue", alpha=0.3)

    # Save figure to file
    plt.savefig("shoreface_{}.png".format(n_obj_plot), dpi=300)
