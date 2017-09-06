from __future__ import division, print_function, absolute_import

import warnings

from distutils.version import LooseVersion
from numpy.testing import suppress_warnings
import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_equal, assert_,
                           assert_allclose, assert_warns)
import pytest
from pytest import raises as assert_raises

from labugr.filters.filters import (normalize, tf2zpk, zpk2tf,
                      BadCoefficients)

class TestTf2zpk(object):

    def test_simple(self):
        z_r = np.array([0.5, -0.5])
        p_r = np.array([1.j / np.sqrt(2), -1.j / np.sqrt(2)])
        # Sort the zeros/poles so that we don't fail the test if the order
        # changes
        z_r.sort()
        p_r.sort()
        b = np.poly(z_r)
        a = np.poly(p_r)

        z, p, k = tf2zpk(b, a)
        z.sort()
        p.sort()
        assert_array_almost_equal(z, z_r)
        assert_array_almost_equal(p, p_r)

    def test_bad_filter(self):
        # Regression test for #651: better handling of badly conditioned
        # filter coefficients.
        with suppress_warnings():
            warnings.simplefilter("error", BadCoefficients)
            assert_raises(BadCoefficients, tf2zpk, [1e-15], [1.0, 1.0])

class TestZpk2Tf(object):

    def test_identity(self):
        """Test the identity transfer function."""
        z = []
        p = []
        k = 1.
        b, a = zpk2tf(z, p, k)
        b_r = np.array([1.])  # desired result
        a_r = np.array([1.])  # desired result
        # The test for the *type* of the return values is a regression
        # test for ticket #1095.  In the case p=[], zpk2tf used to
        # return the scalar 1.0 instead of array([1.0]).
        assert_array_equal(b, b_r)
        assert_(isinstance(b, np.ndarray))
        assert_array_equal(a, a_r)
        assert_(isinstance(a, np.ndarray))

class TestNormalize(object):

    def test_allclose(self):
        """Test for false positive on allclose in normalize() in
        filter_design.py"""
        # Test to make sure the allclose call within signal.normalize does not
        # choose false positives. Then check against a known output from MATLAB
        # to make sure the fix doesn't break anything.

        # These are the coefficients returned from
        #   `[b,a] = cheby1(8, 0.5, 0.048)'
        # in MATLAB. There are at least 15 significant figures in each
        # coefficient, so it makes sense to test for errors on the order of
        # 1e-13 (this can always be relaxed if different platforms have
        # different rounding errors)
        b_matlab = np.array([2.150733144728282e-11, 1.720586515782626e-10,
                             6.022052805239190e-10, 1.204410561047838e-09,
                             1.505513201309798e-09, 1.204410561047838e-09,
                             6.022052805239190e-10, 1.720586515782626e-10,
                             2.150733144728282e-11])
        a_matlab = np.array([1.000000000000000e+00, -7.782402035027959e+00,
                             2.654354569747454e+01, -5.182182531666387e+01,
                             6.334127355102684e+01, -4.963358186631157e+01,
                             2.434862182949389e+01, -6.836925348604676e+00,
                             8.412934944449140e-01])

        # This is the input to signal.normalize after passing through the
        # equivalent steps in signal.iirfilter as was done for MATLAB
        b_norm_in = np.array([1.5543135865293012e-06, 1.2434508692234413e-05,
                              4.3520780422820447e-05, 8.7041560845640893e-05,
                              1.0880195105705122e-04, 8.7041560845640975e-05,
                              4.3520780422820447e-05, 1.2434508692234413e-05,
                              1.5543135865293012e-06])
        a_norm_in = np.array([7.2269025909127173e+04, -5.6242661430467968e+05,
                              1.9182761917308895e+06, -3.7451128364682454e+06,
                              4.5776121393762771e+06, -3.5869706138592605e+06,
                              1.7596511818472347e+06, -4.9409793515707983e+05,
                              6.0799461347219651e+04])

        b_output, a_output = normalize(b_norm_in, a_norm_in)

        # The test on b works for decimal=14 but the one for a does not. For
        # the sake of consistency, both of these are decimal=13. If something
        # breaks on another platform, it is probably fine to relax this lower.
        assert_array_almost_equal(b_matlab, b_output, decimal=13)
        assert_array_almost_equal(a_matlab, a_output, decimal=13)

    def test_errors(self):
        """Test the error cases."""
        # all zero denominator
        assert_raises(ValueError, normalize, [1, 2], 0)

        # denominator not 1 dimensional
        assert_raises(ValueError, normalize, [1, 2], [[1]])

        # numerator too many dimensions
        assert_raises(ValueError, normalize, [[[1, 2]]], 1)


# class TestPrototypeType(object):

#     def test_output_type(self):
#         # Prototypes should consistently output arrays, not lists
#         # https://github.com/scipy/scipy/pull/441
#         for func in (buttap,
#                      besselap,
#                      lambda N: cheb1ap(N, 1),
#                      lambda N: cheb2ap(N, 20),
#                      lambda N: ellipap(N, 1, 20)):
#             for N in range(7):
#                 z, p, k = func(N)
#                 assert_(isinstance(z, np.ndarray))
#                 assert_(isinstance(p, np.ndarray))
