from __future__ import division, print_function, absolute_import

import sys
import math
import numpy as np
from numpy import sqrt, cos, sin, arctan, exp, log, pi, Inf
from numpy.testing import (assert_,
        assert_allclose, assert_array_less, assert_almost_equal, assert_raises)
import pytest

from labugr.integrate import quad, dblquad, tplquad, nquad
from labugr.dependencias.six import xrange
from .._lib._ccallback import LowLevelCallable

import ctypes
import ctypes.util
from .._lib._ccallback_c import sine_ctypes

import labugr.integrate._test_multivariate as clib_test


def assert_quad(value_and_err, tabled_value, errTol=1.5e-8):
    value, err = value_and_err
    assert_allclose(value, tabled_value, atol=err, rtol=0)
    if errTol is not None:
        assert_array_less(err, errTol)


class TestCtypesQuad(object):
    def setup_method(self):
        if sys.platform == 'win32':
            if sys.version_info < (3, 5):
                files = [ctypes.util.find_msvcrt()]
            else:
                files = ['api-ms-win-crt-math-l1-1-0.dll']
        elif sys.platform == 'darwin':
            files = ['libm.dylib']
        else:
            files = ['libm.so', 'libm.so.6']

        for file in files:
            try:
                self.lib = ctypes.CDLL(file)
                break
            except OSError:
                pass
        else:
            # This test doesn't work on some Linux platforms (Fedora for
            # example) that put an ld script in libm.so - see gh-5370
            self.skipTest("Ctypes can't import libm.so")

        restype = ctypes.c_double
        argtypes = (ctypes.c_double,)
        for name in ['sin', 'cos', 'tan']:
            func = getattr(self.lib, name)
            func.restype = restype
            func.argtypes = argtypes

    def test_typical(self):
        assert_quad(quad(self.lib.sin, 0, 5), quad(math.sin, 0, 5)[0])
        assert_quad(quad(self.lib.cos, 0, 5), quad(math.cos, 0, 5)[0])
        assert_quad(quad(self.lib.tan, 0, 1), quad(math.tan, 0, 1)[0])

    def test_ctypes_sine(self):
        quad(LowLevelCallable(sine_ctypes), 0, 1)

    def test_ctypes_variants(self):
        lib = ctypes.CDLL(clib_test.__file__)

        sin_0 = lib._sin_0
        sin_0.restype = ctypes.c_double
        sin_0.argtypes = [ctypes.c_double, ctypes.c_void_p]

        sin_1 = lib._sin_1
        sin_1.restype = ctypes.c_double
        sin_1.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p]

        sin_2 = lib._sin_2
        sin_2.restype = ctypes.c_double
        sin_2.argtypes = [ctypes.c_double]

        sin_3 = lib._sin_3
        sin_3.restype = ctypes.c_double
        sin_3.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

        sin_4 = lib._sin_3
        sin_4.restype = ctypes.c_double
        sin_4.argtypes = [ctypes.c_int, ctypes.c_double]

        all_sigs = [sin_0, sin_1, sin_2, sin_3, sin_4]
        legacy_sigs = [sin_2, sin_4]
        legacy_only_sigs = [sin_4]

        # LowLevelCallables work for new signatures
        for j, func in enumerate(all_sigs):
            callback = LowLevelCallable(func)
            if func in legacy_only_sigs:
                assert_raises(ValueError, quad, callback, 0, pi)
            else:
                assert_allclose(quad(callback, 0, pi)[0], 2.0)

        # Plain ctypes items work only for legacy signatures
        for j, func in enumerate(legacy_sigs):
            if func in legacy_sigs:
                assert_allclose(quad(func, 0, pi)[0], 2.0)
            else:
                assert_raises(ValueError, quad, func, 0, pi)


class TestMultivariateCtypesQuad(object):
    def setup_method(self):
        self.lib = ctypes.CDLL(clib_test.__file__)
        restype = ctypes.c_double
        argtypes = (ctypes.c_int, ctypes.c_double)
        for name in ['_multivariate_typical', '_multivariate_indefinite',
                     '_multivariate_sin']:
            func = getattr(self.lib, name)
            func.restype = restype
            func.argtypes = argtypes

    def test_typical(self):
        # 1) Typical function with two extra arguments:
        assert_quad(quad(self.lib._multivariate_typical, 0, pi, (2, 1.8)),
                    0.30614353532540296487)

    def test_indefinite(self):
        # 2) Infinite integration limits --- Euler's constant
        assert_quad(quad(self.lib._multivariate_indefinite, 0, Inf),
                    0.577215664901532860606512)

    def test_threadsafety(self):
        # Ensure multivariate ctypes are threadsafe
        def threadsafety(y):
            return y + quad(self.lib._multivariate_sin, 0, 1)[0]
        assert_quad(quad(threadsafety, 0, 1), 0.9596976941318602)


class TestQuad(object):
    def test_typical(self):
        # 1) Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return cos(n*x-z*sin(x))/pi
        assert_quad(quad(myfunc, 0, pi, (2, 1.8)), 0.30614353532540296487)

    def test_indefinite(self):
        # 2) Infinite integration limits --- Euler's constant
        def myfunc(x):           # Euler's constant integrand
            return -exp(-x)*log(x)
        assert_quad(quad(myfunc, 0, Inf), 0.577215664901532860606512)

    def test_singular(self):
        # 3) Singular points in region of integration.
        def myfunc(x):
            if 0 < x < 2.5:
                return sin(x)
            elif 2.5 <= x <= 5.0:
                return exp(-x)
            else:
                return 0.0

        assert_quad(quad(myfunc, 0, 10, points=[2.5, 5.0]),
                    1 - cos(2.5) + exp(-2.5) - exp(-5.0))

    def test_sine_weighted_finite(self):
        # 4) Sine weighted integral (finite limits)
        def myfunc(x, a):
            return exp(a*(x-1))

        ome = 2.0**3.4
        assert_quad(quad(myfunc, 0, 1, args=20, weight='sin', wvar=ome),
                    (20*sin(ome)-ome*cos(ome)+ome*exp(-20))/(20**2 + ome**2))

    def test_sine_weighted_infinite(self):
        # 5) Sine weighted integral (infinite limits)
        def myfunc(x, a):
            return exp(-x*a)

        a = 4.0
        ome = 3.0
        assert_quad(quad(myfunc, 0, Inf, args=a, weight='sin', wvar=ome),
                    ome/(a**2 + ome**2))

    def test_cosine_weighted_infinite(self):
        # 6) Cosine weighted integral (negative infinite limits)
        def myfunc(x, a):
            return exp(x*a)

        a = 2.5
        ome = 2.3
        assert_quad(quad(myfunc, -Inf, 0, args=a, weight='cos', wvar=ome),
                    a/(a**2 + ome**2))

    def test_algebraic_log_weight(self):
        # 6) Algebraic-logarithmic weight.
        def myfunc(x, a):
            return 1/(1+x+2**(-a))

        a = 1.5
        assert_quad(quad(myfunc, -1, 1, args=a, weight='alg',
                         wvar=(-0.5, -0.5)),
                    pi/sqrt((1+2**(-a))**2 - 1))

    def test_cauchypv_weight(self):
        # 7) Cauchy prinicpal value weighting w(x) = 1/(x-c)
        def myfunc(x, a):
            return 2.0**(-a)/((x-1)**2+4.0**(-a))

        a = 0.4
        tabledValue = ((2.0**(-0.4)*log(1.5) -
                        2.0**(-1.4)*log((4.0**(-a)+16) / (4.0**(-a)+1)) -
                        arctan(2.0**(a+2)) -
                        arctan(2.0**a)) /
                       (4.0**(-a) + 1))
        assert_quad(quad(myfunc, 0, 5, args=0.4, weight='cauchy', wvar=2.0),
                    tabledValue, errTol=1.9e-8)

    def test_double_integral(self):
        # 8) Double Integral test
        def simpfunc(y, x):       # Note order of arguments.
            return x+y

        a, b = 1.0, 2.0
        assert_quad(dblquad(simpfunc, a, b, lambda x: x, lambda x: 2*x),
                    5/6.0 * (b**3.0-a**3.0))

    def test_double_integral2(self):
        def func(x0, x1, t0, t1):
            return x0 + x1 + t0 + t1
        g = lambda x: x
        h = lambda x: 2 * x
        args = 1, 2
        assert_quad(dblquad(func, 1, 2, g, h, args=args),35./6 + 9*.5)

    def test_triple_integral(self):
        # 9) Triple Integral test
        def simpfunc(z, y, x, t):      # Note order of arguments.
            return (x+y+z)*t

        a, b = 1.0, 2.0
        assert_quad(tplquad(simpfunc, a, b,
                            lambda x: x, lambda x: 2*x,
                            lambda x, y: x - y, lambda x, y: x + y,
                            (2.,)),
                     2*8/3.0 * (b**4.0 - a**4.0))


class TestNQuad(object):
    def test_fixed_limits(self):
        def func1(x0, x1, x2, x3):
            val = (x0**2 + x1*x2 - x3**3 + np.sin(x0) +
                   (1 if (x0 - 0.2*x3 - 0.5 - 0.25*x1 > 0) else 0))
            return val

        def opts_basic(*args):
            return {'points': [0.2*args[2] + 0.5 + 0.25*args[0]]}

        res = nquad(func1, [[0, 1], [-1, 1], [.13, .8], [-.15, 1]],
                    opts=[opts_basic, {}, {}, {}], full_output=True)
        assert_quad(res[:-1], 1.5267454070738635)
        assert_(res[-1]['neval'] > 0 and res[-1]['neval'] < 4e5) 
        
    def test_variable_limits(self):
        scale = .1

        def func2(x0, x1, x2, x3, t0, t1):
            val = (x0*x1*x3**2 + np.sin(x2) + 1 +
                   (1 if x0 + t1*x1 - t0 > 0 else 0))
            return val

        def lim0(x1, x2, x3, t0, t1):
            return [scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) - 1,
                    scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) + 1]

        def lim1(x2, x3, t0, t1):
            return [scale * (t0*x2 + t1*x3) - 1,
                    scale * (t0*x2 + t1*x3) + 1]

        def lim2(x3, t0, t1):
            return [scale * (x3 + t0**2*t1**3) - 1,
                    scale * (x3 + t0**2*t1**3) + 1]

        def lim3(t0, t1):
            return [scale * (t0 + t1) - 1, scale * (t0 + t1) + 1]

        def opts0(x1, x2, x3, t0, t1):
            return {'points': [t0 - t1*x1]}

        def opts1(x2, x3, t0, t1):
            return {}

        def opts2(x3, t0, t1):
            return {}

        def opts3(t0, t1):
            return {}

        res = nquad(func2, [lim0, lim1, lim2, lim3], args=(0, 0),
                    opts=[opts0, opts1, opts2, opts3])
        assert_quad(res, 25.066666666666663)

    def test_square_separate_ranges_and_opts(self):
        def f(y, x):
            return 1.0

        assert_quad(nquad(f, [[-1, 1], [-1, 1]], opts=[{}, {}]), 4.0)

    def test_square_aliased_ranges_and_opts(self):
        def f(y, x):
            return 1.0

        r = [-1, 1]
        opt = {}
        assert_quad(nquad(f, [r, r], opts=[opt, opt]), 4.0)

    def test_square_separate_fn_ranges_and_opts(self):
        def f(y, x):
            return 1.0

        def fn_range0(*args):
            return (-1, 1)

        def fn_range1(*args):
            return (-1, 1)

        def fn_opt0(*args):
            return {}

        def fn_opt1(*args):
            return {}

        ranges = [fn_range0, fn_range1]
        opts = [fn_opt0, fn_opt1]
        assert_quad(nquad(f, ranges, opts=opts), 4.0)

    def test_square_aliased_fn_ranges_and_opts(self):
        def f(y, x):
            return 1.0

        def fn_range(*args):
            return (-1, 1)

        def fn_opt(*args):
            return {}

        ranges = [fn_range, fn_range]
        opts = [fn_opt, fn_opt]
        assert_quad(nquad(f, ranges, opts=opts), 4.0)

    def test_matching_quad(self):
        def func(x):
            return x**2 + 1

        res, reserr = quad(func, 0, 4)
        res2, reserr2 = nquad(func, ranges=[[0, 4]])
        assert_almost_equal(res, res2)
        assert_almost_equal(reserr, reserr2)

    def test_matching_dblquad(self):
        def func2d(x0, x1):
            return x0**2 + x1**3 - x0 * x1 + 1

        res, reserr = dblquad(func2d, -2, 2, lambda x: -3, lambda x: 3)
        res2, reserr2 = nquad(func2d, [[-3, 3], (-2, 2)])
        assert_almost_equal(res, res2)
        assert_almost_equal(reserr, reserr2)

    def test_matching_tplquad(self):
        def func3d(x0, x1, x2, c0, c1):
            return x0**2 + c0 * x1**3 - x0 * x1 + 1 + c1 * np.sin(x2)

        res = tplquad(func3d, -1, 2, lambda x: -2, lambda x: 2,
                      lambda x, y: -np.pi, lambda x, y: np.pi,
                      args=(2, 3))
        res2 = nquad(func3d, [[-np.pi, np.pi], [-2, 2], (-1, 2)], args=(2, 3))
        assert_almost_equal(res, res2)

    def test_dict_as_opts(self):
        try:
            out = nquad(lambda x, y: x * y, [[0, 1], [0, 1]], opts={'epsrel': 0.0001})
        except(TypeError):
            assert False
