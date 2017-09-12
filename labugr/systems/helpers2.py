""" Classes for interpolating values.
"""
from __future__ import division, print_function, absolute_import


__all__ = ['interp1d']

import functools
import operator

from labugr.integrate._lib._util import _asarray_validated

import numpy as np
from numpy import (array, transpose, searchsorted, atleast_1d, atleast_2d,
                   dot, ravel, poly1d, asarray, intp)

from labugr.integrate._lib.six import xrange

from . import fitpack 
from . import dfitpack #####
from ._bsplines import make_interp_spline


def prod(x):
    """Product of a list of numbers; ~40x faster vs np.prod for Python tuples"""
    if len(x) == 0:
        return 1
    return functools.reduce(operator.mul, x)


def lagrange(x, w):
    r"""
    Return a Lagrange interpolating polynomial.

    Given two 1-D arrays `x` and `w,` returns the Lagrange interpolating
    polynomial through the points ``(x, w)``.

    Warning: This implementation is numerically unstable. Do not expect to
    be able to use more than about 20 points even if they are chosen optimally.

    Parameters
    ----------
    x : array_like
        `x` represents the x-coordinates of a set of datapoints.
    w : array_like
        `w` represents the y-coordinates of a set of datapoints, i.e. f(`x`).

    Returns
    -------
    lagrange : `numpy.poly1d` instance
        The Lagrange interpolating polynomial.
    
    Examples
    --------
    Interpolate :math:`f(x) = x^3` by 3 points.

    >>> from scipy.interpolate import lagrange
    >>> x = np.array([0, 1, 2])
    >>> y = x**3
    >>> poly = lagrange(x, y)
    
    Since there are only 3 points, Lagrange polynomial has degree 2. Explicitly,
    it is given by

    .. math::

        \begin{aligned}
            L(x) &= 1\times \frac{x (x - 2)}{-1} + 8\times \frac{x (x-1)}{2} \\
                 &= x (-2 + 3x)
        \end{aligned}

    >>> from numpy.polynomial.polynomial import Polynomial
    >>> Polynomial(poly).coef
    array([ 3., -2.,  0.])

    """

    M = len(x)
    p = poly1d(0.0)
    for j in xrange(M):
        pt = poly1d(w[j])
        for k in xrange(M):
            if k == j:
                continue
            fac = x[j]-x[k]
            pt *= poly1d([1.0, -x[k]])/fac
        p += pt
    return p


# !! Need to find argument for keeping initialize.  If it isn't
# !! found, get rid of it!


class interp2d(object):
    """
    interp2d(x, y, z, kind='linear', copy=True, bounds_error=False,
             fill_value=nan)

    Interpolate over a 2-D grid.

    `x`, `y` and `z` are arrays of values used to approximate some function
    f: ``z = f(x, y)``. This class returns a function whose call method uses
    spline interpolation to find the value of new points.

    If `x` and `y` represent a regular grid, consider using
    RectBivariateSpline.

    Note that calling `interp2d` with NaNs present in input values results in
    undefined behaviour.

    Methods
    -------
    __call__

    Parameters
    ----------
    x, y : array_like
        Arrays defining the data point coordinates.

        If the points lie on a regular grid, `x` can specify the column
        coordinates and `y` the row coordinates, for example::

          >>> x = [0,1,2];  y = [0,3]; z = [[1,2,3], [4,5,6]]

        Otherwise, `x` and `y` must specify the full coordinates for each
        point, for example::

          >>> x = [0,1,2,0,1,2];  y = [0,0,0,3,3,3]; z = [1,2,3,4,5,6]

        If `x` and `y` are multi-dimensional, they are flattened before use.
    z : array_like
        The values of the function to interpolate at the data points. If
        `z` is a multi-dimensional array, it is flattened before use.  The
        length of a flattened `z` array is either
        len(`x`)*len(`y`) if `x` and `y` specify the column and row coordinates
        or ``len(z) == len(x) == len(y)`` if `x` and `y` specify coordinates
        for each point.
    kind : {'linear', 'cubic', 'quintic'}, optional
        The kind of spline interpolation to use. Default is 'linear'.
    copy : bool, optional
        If True, the class makes internal copies of x, y and z.
        If False, references may be used. The default is to copy.
    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data (x,y), a ValueError is raised.
        If False, then `fill_value` is used.
    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If omitted (None), values outside
        the domain are extrapolated.

    See Also
    --------
    RectBivariateSpline :
        Much faster 2D interpolation if your input data is on a grid
    bisplrep, bisplev :
        Spline interpolation based on FITPACK
    BivariateSpline : a more recent wrapper of the FITPACK routines
    interp1d : one dimension version of this function

    Notes
    -----
    The minimum number of data points required along the interpolation
    axis is ``(k+1)**2``, with k=1 for linear, k=3 for cubic and k=5 for
    quintic interpolation.

    The interpolator is constructed by `bisplrep`, with a smoothing factor
    of 0. If more control over smoothing is needed, `bisplrep` should be
    used directly.

    Examples
    --------
    Construct a 2-D grid and interpolate on it:

    >>> from scipy import interpolate
    >>> x = np.arange(-5.01, 5.01, 0.25)
    >>> y = np.arange(-5.01, 5.01, 0.25)
    >>> xx, yy = np.meshgrid(x, y)
    >>> z = np.sin(xx**2+yy**2)
    >>> f = interpolate.interp2d(x, y, z, kind='cubic')

    Now use the obtained interpolation function and plot the result:

    >>> import matplotlib.pyplot as plt
    >>> xnew = np.arange(-5.01, 5.01, 1e-2)
    >>> ynew = np.arange(-5.01, 5.01, 1e-2)
    >>> znew = f(xnew, ynew)
    >>> plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
    >>> plt.show()
    """

    def __init__(self, x, y, z, kind='linear', copy=True, bounds_error=False,
                 fill_value=None):
        x = ravel(x)
        y = ravel(y)
        z = asarray(z)

        rectangular_grid = (z.size == len(x) * len(y))
        if rectangular_grid:
            if z.ndim == 2:
                if z.shape != (len(y), len(x)):
                    raise ValueError("When on a regular grid with x.size = m "
                                     "and y.size = n, if z.ndim == 2, then z "
                                     "must have shape (n, m)")
            if not np.all(x[1:] >= x[:-1]):
                j = np.argsort(x)
                x = x[j]
                z = z[:, j]
            if not np.all(y[1:] >= y[:-1]):
                j = np.argsort(y)
                y = y[j]
                z = z[j, :]
            z = ravel(z.T)
        else:
            z = ravel(z)
            if len(x) != len(y):
                raise ValueError(
                    "x and y must have equal lengths for non rectangular grid")
            if len(z) != len(x):
                raise ValueError(
                    "Invalid length for input z for non rectangular grid")

        try:
            kx = ky = {'linear': 1,
                       'cubic': 3,
                       'quintic': 5}[kind]
        except KeyError:
            raise ValueError("Unsupported interpolation type.")

        if not rectangular_grid:
            # TODO: surfit is really not meant for interpolation!
            self.tck = fitpack.bisplrep(x, y, z, kx=kx, ky=ky, s=0.0)
        else:
            nx, tx, ny, ty, c, fp, ier = dfitpack.regrid_smth(
                x, y, z, None, None, None, None,
                kx=kx, ky=ky, s=0.0)
            self.tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)],
                        kx, ky)

        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.x, self.y, self.z = [array(a, copy=copy) for a in (x, y, z)]

        self.x_min, self.x_max = np.amin(x), np.amax(x)
        self.y_min, self.y_max = np.amin(y), np.amax(y)

    def __call__(self, x, y, dx=0, dy=0, assume_sorted=False):
        """Interpolate the function.

        Parameters
        ----------
        x : 1D array
            x-coordinates of the mesh on which to interpolate.
        y : 1D array
            y-coordinates of the mesh on which to interpolate.
        dx : int >= 0, < kx
            Order of partial derivatives in x.
        dy : int >= 0, < ky
            Order of partial derivatives in y.
        assume_sorted : bool, optional
            If False, values of `x` and `y` can be in any order and they are
            sorted first.
            If True, `x` and `y` have to be arrays of monotonically
            increasing values.

        Returns
        -------
        z : 2D array with shape (len(y), len(x))
            The interpolated values.
        """

        x = atleast_1d(x)
        y = atleast_1d(y)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y should both be 1-D arrays")

        if not assume_sorted:
            x = np.sort(x)
            y = np.sort(y)

        if self.bounds_error or self.fill_value is not None:
            out_of_bounds_x = (x < self.x_min) | (x > self.x_max)
            out_of_bounds_y = (y < self.y_min) | (y > self.y_max)

            any_out_of_bounds_x = np.any(out_of_bounds_x)
            any_out_of_bounds_y = np.any(out_of_bounds_y)

        if self.bounds_error and (any_out_of_bounds_x or any_out_of_bounds_y):
            raise ValueError("Values out of range; x must be in %r, y in %r"
                             % ((self.x_min, self.x_max),
                                (self.y_min, self.y_max)))

        z = fitpack.bisplev(x, y, self.tck, dx, dy)
        z = atleast_2d(z)
        z = transpose(z)

        if self.fill_value is not None:
            if any_out_of_bounds_x:
                z[:, out_of_bounds_x] = self.fill_value
            if any_out_of_bounds_y:
                z[out_of_bounds_y, :] = self.fill_value

        if len(z) == 1:
            z = z[0]
        return array(z)


def _check_broadcast_up_to(arr_from, shape_to, name):
    """Helper to check that arr_from broadcasts up to shape_to"""
    shape_from = arr_from.shape
    if len(shape_to) >= len(shape_from):
        for t, f in zip(shape_to[::-1], shape_from[::-1]):
            if f != 1 and f != t:
                break
        else:  # all checks pass, do the upcasting that we need later
            if arr_from.size != 1 and arr_from.shape != shape_to:
                arr_from = np.ones(shape_to, arr_from.dtype) * arr_from
            return arr_from.ravel()
    # at least one check failed
    raise ValueError('%s argument must be able to broadcast up '
                     'to shape %s but had shape %s'
                     % (name, shape_to, shape_from))


def _do_extrapolate(fill_value):
    """Helper to check if fill_value == "extrapolate" without warnings"""
    return (isinstance(fill_value, string_types) and
            fill_value == 'extrapolate')


class interp1d(_Interpolator1D):
    """
    Interpolate a 1-D function.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``.  This class returns a function whose call method uses
    interpolation to find the value of new points.

    Note that calling `interp1d` with NaNs present in input values results in
    undefined behaviour.

    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    y : (...,N,...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order) or as an
        integer specifying the order of the spline interpolator to use.
        Default is 'linear'.
    axis : int, optional
        Specifies the axis of `y` along which to interpolate.
        Interpolation defaults to the last axis of `y`.
    copy : bool, optional
        If True, the class makes internal copies of x and y.
        If False, references to `x` and `y` are used. The default is to copy.
    bounds_error : bool, optional
        If True, a ValueError is raised any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised unless `fill_value="extrapolate"`.
    fill_value : array-like or (array-like, array_like) or "extrapolate", optional
        - if a ndarray (or float), this value will be used to fill in for
          requested points outside of the data range. If not provided, then
          the default is NaN. The array-like must broadcast properly to the
          dimensions of the non-interpolation axes.
        - If a two-element tuple, then the first element is used as a
          fill value for ``x_new < x[0]`` and the second element is used for
          ``x_new > x[-1]``. Anything that is not a 2-element tuple (e.g.,
          list or ndarray, regardless of shape) is taken to be a single
          array-like argument meant to be used for both bounds as
          ``below, above = fill_value, fill_value``.

          .. versionadded:: 0.17.0
        - If "extrapolate", then points outside the data range will be
          extrapolated.

          .. versionadded:: 0.17.0
    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.

    Methods
    -------
    __call__

    See Also
    --------
    splrep, splev
        Spline interpolation/smoothing based on FITPACK.
    UnivariateSpline : An object-oriented wrapper of the FITPACK routines.
    interp2d : 2-D interpolation

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy import interpolate
    >>> x = np.arange(0, 10)
    >>> y = np.exp(-x/3.0)
    >>> f = interpolate.interp1d(x, y)

    >>> xnew = np.arange(0, 9, 0.1)
    >>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
    >>> plt.plot(x, y, 'o', xnew, ynew, '-')
    >>> plt.show()
    """

    def __init__(self, x, y, kind='linear', axis=-1,
                 copy=True, bounds_error=None, fill_value=np.nan,
                 assume_sorted=False):
        """ Initialize a 1D linear interpolation class."""
        _Interpolator1D.__init__(self, x, y, axis=axis)

        self.bounds_error = bounds_error  # used by fill_value setter
        self.copy = copy

        if kind in ['zero', 'slinear', 'quadratic', 'cubic']:
            order = {'zero': 0, 'slinear': 1,
                     'quadratic': 2, 'cubic': 3}[kind]
            kind = 'spline'
        elif isinstance(kind, int):
            order = kind
            kind = 'spline'
        elif kind not in ('linear', 'nearest'):
            raise NotImplementedError("%s is unsupported: Use fitpack "
                                      "routines for other types." % kind)
        x = array(x, copy=self.copy)
        y = array(y, copy=self.copy)

        if not assume_sorted:
            ind = np.argsort(x)
            x = x[ind]
            y = np.take(y, ind, axis=axis)

        if x.ndim != 1:
            raise ValueError("the x array must have exactly one dimension.")
        if y.ndim == 0:
            raise ValueError("the y array must have at least one dimension.")

        # Force-cast y to a floating-point type, if it's not yet one
        if not issubclass(y.dtype.type, np.inexact):
            y = y.astype(np.float_)

        # Backward compatibility
        self.axis = axis % y.ndim

        # Interpolation goes internally along the first axis
        self.y = y
        self._y = self._reshape_yi(self.y)
        self.x = x
        del y, x  # clean up namespace to prevent misuse; use attributes
        self._kind = kind
        self.fill_value = fill_value  # calls the setter, can modify bounds_err

        # Adjust to interpolation kind; store reference to *unbound*
        # interpolation methods, in order to avoid circular references to self
        # stored in the bound instance methods, and therefore delayed garbage
        # collection.  See: http://docs.python.org/2/reference/datamodel.html
        if kind in ('linear', 'nearest'):
            # Make a "view" of the y array that is rotated to the interpolation
            # axis.
            minval = 2
            if kind == 'nearest':
                # Do division before addition to prevent possible integer
                # overflow
                self.x_bds = self.x / 2.0
                self.x_bds = self.x_bds[1:] + self.x_bds[:-1]

                self._call = self.__class__._call_nearest
            else:
                # Check if we can delegate to numpy.interp (2x-10x faster).
                cond = self.x.dtype == np.float_ and self.y.dtype == np.float_
                cond = cond and self.y.ndim == 1
                cond = cond and not _do_extrapolate(fill_value)

                if cond:
                    self._call = self.__class__._call_linear_np
                else:
                    self._call = self.__class__._call_linear
        else:
            minval = order + 1

            rewrite_nan = False
            xx, yy = self.x, self._y
            if order > 1:
                # Quadratic or cubic spline. If input contains even a single
                # nan, then the output is all nans. We cannot just feed data
                # with nans to make_interp_spline because it calls LAPACK.
                # So, we make up a bogus x and y with no nans and use it
                # to get the correct shape of the output, which we then fill
                # with nans.
                # For slinear or zero order spline, we just pass nans through.
                if np.isnan(self.x).any():
                    xx = np.linspace(min(self.x), max(self.x), len(self.x))
                    rewrite_nan = True
                if np.isnan(self._y).any():
                    yy = np.ones_like(self._y)
                    rewrite_nan = True

            self._spline = make_interp_spline(xx, yy, k=order,
                                              check_finite=False)
            if rewrite_nan:
                self._call = self.__class__._call_nan_spline
            else:
                self._call = self.__class__._call_spline

        if len(self.x) < minval:
            raise ValueError("x and y arrays must have at "
                             "least %d entries" % minval)

    @property
    def fill_value(self):
        # backwards compat: mimic a public attribute
        return self._fill_value_orig

    @fill_value.setter
    def fill_value(self, fill_value):
        # extrapolation only works for nearest neighbor and linear methods
        if _do_extrapolate(fill_value):
            if self.bounds_error:
                raise ValueError("Cannot extrapolate and raise "
                                 "at the same time.")
            self.bounds_error = False
            self._extrapolate = True
        else:
            broadcast_shape = (self.y.shape[:self.axis] +
                               self.y.shape[self.axis + 1:])
            if len(broadcast_shape) == 0:
                broadcast_shape = (1,)
            # it's either a pair (_below_range, _above_range) or a single value
            # for both above and below range
            if isinstance(fill_value, tuple) and len(fill_value) == 2:
                below_above = [np.asarray(fill_value[0]),
                               np.asarray(fill_value[1])]
                names = ('fill_value (below)', 'fill_value (above)')
                for ii in range(2):
                    below_above[ii] = _check_broadcast_up_to(
                        below_above[ii], broadcast_shape, names[ii])
            else:
                fill_value = np.asarray(fill_value)
                below_above = [_check_broadcast_up_to(
                    fill_value, broadcast_shape, 'fill_value')] * 2
            self._fill_value_below, self._fill_value_above = below_above
            self._extrapolate = False
            if self.bounds_error is None:
                self.bounds_error = True
        # backwards compat: fill_value was a public attr; make it writeable
        self._fill_value_orig = fill_value

    def _call_linear_np(self, x_new):
        # Note that out-of-bounds values are taken care of in self._evaluate
        return np.interp(x_new, self.x, self.y)

    def _call_linear(self, x_new):
        # 2. Find where in the orignal data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
        x_new_indices = searchsorted(self.x, x_new)

        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1.  Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = x_new_indices.clip(1, len(self.x)-1).astype(int)

        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self._y[lo]
        y_hi = self._y[hi]

        # Note that the following two expressions rely on the specifics of the
        # broadcasting semantics.
        slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

        # 5. Calculate the actual value for each entry in x_new.
        y_new = slope*(x_new - x_lo)[:, None] + y_lo

        return y_new

    def _call_nearest(self, x_new):
        """ Find nearest neighbour interpolated y_new = f(x_new)."""

        # 2. Find where in the averaged data the values to interpolate
        #    would be inserted.
        #    Note: use side='left' (right) to searchsorted() to define the
        #    halfway point to be nearest to the left (right) neighbour
        x_new_indices = searchsorted(self.x_bds, x_new, side='left')

        # 3. Clip x_new_indices so that they are within the range of x indices.
        x_new_indices = x_new_indices.clip(0, len(self.x)-1).astype(intp)

        # 4. Calculate the actual value for each entry in x_new.
        y_new = self._y[x_new_indices]

        return y_new

    def _call_spline(self, x_new):
        return self._spline(x_new)

    def _call_nan_spline(self, x_new):
        out = self._spline(x_new)
        out[...] = np.nan
        return out

    def _evaluate(self, x_new):
        # 1. Handle values in x_new that are outside of x.  Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        x_new = asarray(x_new)
        y_new = self._call(self, x_new)
        if not self._extrapolate:
            below_bounds, above_bounds = self._check_bounds(x_new)
            if len(y_new) > 0:
                # Note fill_value must be broadcast up to the proper size
                # and flattened to work here
                y_new[below_bounds] = self._fill_value_below
                y_new[above_bounds] = self._fill_value_above
        return y_new

    def _check_bounds(self, x_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array

        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """

        # If self.bounds_error is True, we raise an error if any x_new values
        # fall outside the range of x.  Otherwise, we return an array indicating
        # which values are outside the boundary region.
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        # !! Could provide more information about which values are out of bounds
        if self.bounds_error and below_bounds.any():
            raise ValueError("A value in x_new is below the interpolation "
                             "range.")
        if self.bounds_error and above_bounds.any():
            raise ValueError("A value in x_new is above the interpolation "
                             "range.")

        # !! Should we emit a warning if some values are out of bounds?
        # !! matlab does not.
        return below_bounds, above_bounds

class _Interpolator1D(object):
    """
    Common features in univariate interpolation

    Deal with input data type and interpolation axis rolling.  The
    actual interpolator can assume the y-data is of shape (n, r) where
    `n` is the number of x-points, and `r` the number of variables,
    and use self.dtype as the y-data type.

    Attributes
    ----------
    _y_axis
        Axis along which the interpolation goes in the original array
    _y_extra_shape
        Additional trailing shape of the input arrays, excluding
        the interpolation axis.
    dtype
        Dtype of the y-data arrays. Can be set via set_dtype, which
        forces it to be float or complex.

    Methods
    -------
    __call__
    _prepare_x
    _finish_y
    _reshape_yi
    _set_yi
    _set_dtype
    _evaluate

    """

    __slots__ = ('_y_axis', '_y_extra_shape', 'dtype')

    def __init__(self, xi=None, yi=None, axis=None):
        self._y_axis = axis
        self._y_extra_shape = None
        self.dtype = None
        if yi is not None:
            self._set_yi(yi, xi=xi, axis=axis)

    def __call__(self, x):
        """
        Evaluate the interpolant

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate(x)
        return self._finish_y(y, x_shape)

    def _evaluate(self, x):
        """
        Actually evaluate the value of the interpolator.
        """
        raise NotImplementedError()

    def _prepare_x(self, x):
        """Reshape input x array to 1-D"""
        x = _asarray_validated(x, check_finite=False, as_inexact=True)
        x_shape = x.shape
        return x.ravel(), x_shape

    def _finish_y(self, y, x_shape):
        """Reshape interpolated y back to n-d array similar to initial y"""
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = (list(range(nx, nx + self._y_axis))
                 + list(range(nx)) + list(range(nx+self._y_axis, nx+ny)))
            y = y.transpose(s)
        return y

    def _reshape_yi(self, yi, check=False):
        yi = np.rollaxis(np.asarray(yi), self._y_axis)
        if check and yi.shape[1:] != self._y_extra_shape:
            ok_shape = "%r + (N,) + %r" % (self._y_extra_shape[-self._y_axis:],
                                           self._y_extra_shape[:-self._y_axis])
            raise ValueError("Data must be of shape %s" % ok_shape)
        return yi.reshape((yi.shape[0], -1))

    def _set_yi(self, yi, xi=None, axis=None):
        if axis is None:
            axis = self._y_axis
        if axis is None:
            raise ValueError("no interpolation axis specified")

        yi = np.asarray(yi)

        shape = yi.shape
        if shape == ():
            shape = (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError("x and y arrays must be equal in length along "
                             "interpolation axis.")

        self._y_axis = (axis % yi.ndim)
        self._y_extra_shape = yi.shape[:self._y_axis]+yi.shape[self._y_axis+1:]
        self.dtype = None
        self._set_dtype(yi.dtype)

    def _set_dtype(self, dtype, union=False):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.dtype, np.complexfloating):
            self.dtype = np.complex_
        else:
            if not union or self.dtype != np.complex_:
                self.dtype = np.float_