import numpy as np
from xtgeo.surface import RegularSurface
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("TkAgg")


rate = 100.0


def bathy(x):
    """ Shape of initial shoreface/seafloor """
    return np.maximum(np.minimum(1.0, np.exp(-x/rate)), 0.00)


def bathy2d(x, y):
    return bathy(y)


def foreshore_profile(x):
    """ Clinoform shape """
    return np.exp(-x/rate)


class Section:
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
        elif x < self.x_left_:
            return self.z_left_


class Bedset3D:
    def __init__(self, strike_limits, dip_limits, n_strike, n_dip, bathymetry, shoreline_position, shoreline_displacement):
        self.x_min_ = strike_limits[0]
        self.x_max_ = strike_limits[1]
        self.y_min_ = dip_limits[0]
        self.y_max_ = dip_limits[1]
        self.nx_ = n_strike
        self.ny_ = n_dip
        self.bottom_ = bathymetry
        self.shore_ = shoreline_position
        self.delta_y_ = shoreline_displacement[0]
        self.delta_z_ = shoreline_displacement[1]
        self.strike_ = np.linspace(self.x_min_, self.x_max_, self.nx_)
        self.sections_ = []
        for i, x in enumerate(self.strike_):
            yy = np.linspace(self.shore_[i], self.y_max_, self.ny_)
            local_bathymetry = np.array([self.bottom_(x, y) for y in yy])
            c = Section(self.shore_[i], self.y_max_, local_bathymetry, self.delta_y_[i], self.delta_z_[i])
            self.sections_.append(c)
    
    def get_top(self):
        """ Assemble arrays of x, y and z coordinates of top surface """
        xa = np.empty(shape=(self.nx_, self.ny_))
        ya = np.empty_like(xa)
        za = np.empty_like(xa)

        for i, x in enumerate(self.strike_):
            yy = np.linspace(self.shore_[i], self.y_max_, self.ny_)
            for j, y in enumerate(yy):
                xa[i, j] = x
                ya[i, j] = y
                za[i, j] = self.sections_[i].get_top(y)

        return xa, ya, za

    def get_top_regular(self):
        """ Assemble arrays of x, y and z coordinates of top surface """
        xa = np.empty(shape=(self.nx_, self.ny_))
        ya = np.empty_like(xa)
        za = np.empty_like(xa)

        for i, x in enumerate(self.strike_):
            yy = np.linspace(self.y_min_, self.y_max_, self.ny_)
            for j, y in enumerate(yy):
                xa[i, j] = x
                ya[i, j] = y
                za[i, j] = self.sections_[i].get_top(y)

        return xa, ya, za

    def evaluate_top(self, x, y): # TODO: Improve this interpolation (remember - only need to evaluate at grid points)
        """ Evaluate top surface at a point """
        xinc = (self.x_max_ - self.x_min_) / (self.nx_ - 1)
        ia = int(np.floor(x / xinc))
        ib = int(np.ceil(x / xinc))
        if ib == ia:
            return self.sections_[ia].get_top(y)
        else:
            wa = (x - ia * xinc) / xinc
            wb = (ib * xinc - x) / xinc
            za = self.sections_[ia].get_top(y)
            zb = self.sections_[ib].get_top(y)
            if za is not None and zb is not None:
                return wa * za + wb * zb
            elif za is None and zb is not None:
                return zb
            elif zb is None and za is not None:
                return za
            else:
                return np.nan

    
    def get_bottom(self):
        """ Assemble arrays of x, y and z coordinates of bottom surface """
        raise NotImplementedError


# Test
xmin, xmax = 0.0, 2000.0
ymin, ymax = 0.0, 1000.0
nx, ny = 80, 100

zb = np.empty(shape=(nx, ny))
xx = np.linspace(xmin, xmax, nx)
yy = np.linspace(ymin, ymax, ny)

for i, x in enumerate(xx):
    for j, y in enumerate(yy):
        zb[i, j] = bathy2d(x, y)

xm, ym = np.meshgrid(xx, yy)

xinc_grid = (xmax - xmin)/(nx-1)
yinc_grid = (ymax - ymin)/(ny-1)

dy = np.array([30.0 * np.exp(-((x - 900.0)/500.0)**2) for x in xx])
dz = np.full(shape=(nx,), fill_value=0.01)
y0 = np.zeros(shape=(nx,))

n_bs = 10

bedsets = []
current_bathymetry = bathy2d

for t in range(n_bs):
    bs = Bedset3D([xmin, xmax], [ymin, ymax], nx, ny, current_bathymetry, y0, (dy, dz))
    bedsets.append(bs)
    y0 += dy
    current_bathymetry = bedsets[-1].evaluate_top
    x_bs, y_bs, z_bs = bs.get_top_regular()
    surf = RegularSurface(
        ncol=nx, nrow=ny, xori=0.0, yori=0.0, xinc=xinc_grid, yinc=yinc_grid,
        rotation=0.0, values=z_bs, yflip=1,
    )
    surf.to_file("event_{}.IRAPG".format(t), "irap_ascii")

surf = RegularSurface(
        ncol=nx, nrow=ny, xori=0.0, yori=0.0, xinc=xinc_grid, yinc=yinc_grid,
        rotation=0.0, values=zb, yflip=1,
    )
surf.to_file("bottom.IRAPG", "irap_ascii")







