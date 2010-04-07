#!/usr/bin/env python
"""Define the ND Potential Energy Surface (PES) class"""

import numpy as np
from scipy.interpolate import RectBivariateSpline as rbs
import scipy.interpolate.fitpack as fitpack
from pardyn.tools import spline, splineeval

class LinearInterpolator:
    def __init__(self, x, y):
        self.x = x.copy()
        self.y = y.copy()

    def get_value(self, x):
        if not self.x[0] <= x <= self.x[-1]:
            raise ValueError

        i0 = np.nonzero(x >= self.x)[0][-1]
        i1 =  np.nonzero(x <=self.x)[0][0]
        if i0 == i1:
            return self.y[i0]
        else:
            t = (x - self.x[i0]) / (self.x[i1] - self.x[i0])
            y = self.y[i0] + t * (self.y[i1] - self.y[i0])
            return y

    def get_derivative(self, x, h=1e-5):
        if not self.x[0] <= x <= self.x[-1]:
            raise ValueError

        if (x + h) > self.x[-1]:
            dydx = (self.get_value(x) - self.get_value(x - h)) / h
        elif (x - h) < self.x[0]:
            dydx = (self.get_value(x + h) - self.get_value(x)) / h
        else:
            dydx = (self.get_value(x + h) - self.get_value(x - h)) / (2.0 * h)

        return dydx

class SplineInterpolator:
    def __init__(self, x, y):
        spl = spline(x, y)
        self.x = x
        self.spl = spl

    def get_value(self, x):
        y = splineeval(self.x, self.spl, np.asarray(x))
        return y[0]

    def get_derivative(self, x, h=1e-5):
        dydx = (self.get_value(x + h) - self.get_value(x - h)) / (2.0 * h)
        return dydx

class PES1D:
    """1D potential energy surface"""
    def __init__(self, x, y, interpolation='spline'):
        if interpolation == 'spline':
            self.interpolator = SplineInterpolator(x, y)
        elif interpolation == 'linear':
            self.interpolator = LinearInterpolator(x, y)
        else:
            raise NotImplementedError

    def get_value(self, x):
        """Evaluate value at x"""
        return self.interpolator.get_value(x)

    def get_derivative(self, x, **kwargs):
        """Evaluate derivative at x"""
        return self.interpolator.get_derivative(x, **kwargs)

class PES2D:
    """2D potential energy surface"""
    def __init__(self, x, y, k=2):
        self.kx0, self.kx1 = k, k
        self.rbs = rbs(x[0], x[1], y, kx=k, ky=k)

    def get_value(self, x):
        """Evaluate value at x"""
        return self.rbs.ev(x[0], x[1])[0]

    def get_derivative(self, x, dx=None):
        """Evaluate derivative at x"""
        if dx is None:
            dx = [0, 0]

        knots = self.rbs.get_knots()
        coeffs = self.rbs.get_coeffs()
        tck = [knots[0], knots[1], coeffs, self.kx0, self.kx1]
        return fitpack.bisplev(x[0], x[1], tck, dx[0], dx[1])

class PES:
    """Class describing ND potential energy surface"""
    def __init__(self, x, y, dim=None, symm=None, **kwargs):
        x = np.asarray(x)
        if dim is None:
            dim = x.reshape(-1, x.shape[-1]).shape[0]
        self.dim = dim

        if dim == 1:
            self.pes = PES1D(x, y, **kwargs)
        elif dim == 2:
            self.pes = PES2D(x, y, **kwargs)
        else:
            raise NotImplementedError, "Dimensionality must be 1 or 2"

        if symm is None:
            def dummy_symm(x):
                """Define no symmetries"""
                return x
            symm = dummy_symm

        self.symm = symm

    def get_value(self, x, **kwargs):
        """Evaluate value at x"""
        x = np.asarray(x)
        #x = self.symm(x)
        return self.pes.get_value(x, **kwargs)

    def get_derivative(self, x, **kwargs):
        """Evaluate derivative at x"""
        x = np.asarray(x)
        """
        symmx = self.symm(x)

        dx = kwargs.get('dx', None)
        print x, symmx
        if (symmx != x).any() and dx is not None:
            idx = np.nonzero(dx)[0][0]
            s = np.sign(x[idx]/symmx[idx])
            return s * self.pes.get_derivative(x, **kwargs)
        """
        return self.pes.get_derivative(x, **kwargs)
