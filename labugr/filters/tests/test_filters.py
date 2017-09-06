from __future__ import division, print_function, absolute_import

import warnings

from distutils.version import LooseVersion
from numpy.testing import suppress_warnings
import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_equal, assert_,
                           assert_allclose, assert_warns)
from numpy.testing import assert_almost_equal
import pytest
from pytest import raises as assert_raises

from labugr.filters.filters import (normalize, tf2zpk, zpk2tf,
                      BadCoefficients, lfilter)
from labugr.filters.iir_filters import butter
from decimal import Decimal

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



class _TestLinearFilter(object):
    def generate(self, shape):
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        return self.convert_dtype(x)

    def convert_dtype(self, arr):
        if self.dtype == np.dtype('O'):
            arr = np.asarray(arr)
            out = np.empty(arr.shape, self.dtype)
            iter = np.nditer([arr, out], ['refs_ok','zerosize_ok'],
                        [['readonly'],['writeonly']])
            for x, y in iter:
                y[...] = self.type(x[()])
            return out
        else:
            return np.array(arr, self.dtype, copy=False)

    def test_rank_1_IIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, -0.5])
        y_r = self.convert_dtype([0, 2, 4, 6, 8, 10.])
        assert_array_almost_equal(lfilter(b, a, x), y_r)

    def test_rank_1_FIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 1])
        a = self.convert_dtype([1])
        y_r = self.convert_dtype([0, 1, 3, 5, 7, 9.])
        assert_array_almost_equal(lfilter(b, a, x), y_r)

    def test_rank_1_IIR_init_cond(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([0.5, -0.5])
        zi = self.convert_dtype([1, 2])
        y_r = self.convert_dtype([1, 5, 9, 13, 17, 21])
        zf_r = self.convert_dtype([13, -10])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, y_r)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_1_FIR_init_cond(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 1, 1])
        a = self.convert_dtype([1])
        zi = self.convert_dtype([1, 1])
        y_r = self.convert_dtype([1, 2, 3, 6, 9, 12.])
        zf_r = self.convert_dtype([9, 5])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, y_r)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_2_IIR_axis_0(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        y_r2_a0 = self.convert_dtype([[0, 2, 4], [6, 4, 2], [0, 2, 4],
                                      [6, 4, 2]])
        y = lfilter(b, a, x, axis=0)
        assert_array_almost_equal(y_r2_a0, y)

    def test_rank_2_IIR_axis_1(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        y_r2_a1 = self.convert_dtype([[0, 2, 0], [6, -4, 6], [12, -10, 12],
                            [18, -16, 18]])
        y = lfilter(b, a, x, axis=1)
        assert_array_almost_equal(y_r2_a1, y)

    def test_rank_2_IIR_axis_0_init_cond(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        zi = self.convert_dtype(np.ones((4,1)))

        y_r2_a0_1 = self.convert_dtype([[1, 1, 1], [7, -5, 7], [13, -11, 13],
                              [19, -17, 19]])
        zf_r = self.convert_dtype([-5, -17, -29, -41])[:, np.newaxis]
        y, zf = lfilter(b, a, x, axis=1, zi=zi)
        assert_array_almost_equal(y_r2_a0_1, y)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_2_IIR_axis_1_init_cond(self):
        x = self.generate((4,3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        zi = self.convert_dtype(np.ones((1,3)))

        y_r2_a0_0 = self.convert_dtype([[1, 3, 5], [5, 3, 1],
                                        [1, 3, 5], [5, 3, 1]])
        zf_r = self.convert_dtype([[-23, -23, -23]])
        y, zf = lfilter(b, a, x, axis=0, zi=zi)
        assert_array_almost_equal(y_r2_a0_0, y)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_3_IIR(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])

        for axis in range(x.ndim):
            y = lfilter(b, a, x, axis)
            y_r = np.apply_along_axis(lambda w: lfilter(b, a, w), axis, x)
            assert_array_almost_equal(y, y_r)

    def test_rank_3_IIR_init_cond(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])

        for axis in range(x.ndim):
            zi_shape = list(x.shape)
            zi_shape[axis] = 1
            zi = self.convert_dtype(np.ones(zi_shape))
            zi1 = self.convert_dtype([1])
            y, zf = lfilter(b, a, x, axis, zi)
            lf0 = lambda w: lfilter(b, a, w, zi=zi1)[0]
            lf1 = lambda w: lfilter(b, a, w, zi=zi1)[1]
            y_r = np.apply_along_axis(lf0, axis, x)
            zf_r = np.apply_along_axis(lf1, axis, x)
            assert_array_almost_equal(y, y_r)
            assert_array_almost_equal(zf, zf_r)

    def test_rank_3_FIR(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([1])

        for axis in range(x.ndim):
            y = lfilter(b, a, x, axis)
            y_r = np.apply_along_axis(lambda w: lfilter(b, a, w), axis, x)
            assert_array_almost_equal(y, y_r)

    def test_rank_3_FIR_init_cond(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([1])

        for axis in range(x.ndim):
            zi_shape = list(x.shape)
            zi_shape[axis] = 2
            zi = self.convert_dtype(np.ones(zi_shape))
            zi1 = self.convert_dtype([1, 1])
            y, zf = lfilter(b, a, x, axis, zi)
            lf0 = lambda w: lfilter(b, a, w, zi=zi1)[0]
            lf1 = lambda w: lfilter(b, a, w, zi=zi1)[1]
            y_r = np.apply_along_axis(lf0, axis, x)
            zf_r = np.apply_along_axis(lf1, axis, x)
            assert_array_almost_equal(y, y_r)
            assert_array_almost_equal(zf, zf_r)

    def test_zi_pseudobroadcast(self):
        x = self.generate((4, 5, 20))
        b,a = butter(8, 0.2, output='ba')
        b = self.convert_dtype(b)
        a = self.convert_dtype(a)
        zi_size = b.shape[0] - 1

        # lfilter requires x.ndim == zi.ndim exactly.  However, zi can have
        # length 1 dimensions.
        zi_full = self.convert_dtype(np.ones((4, 5, zi_size)))
        zi_sing = self.convert_dtype(np.ones((1, 1, zi_size)))

        y_full, zf_full = lfilter(b, a, x, zi=zi_full)
        y_sing, zf_sing = lfilter(b, a, x, zi=zi_sing)

        assert_array_almost_equal(y_sing, y_full)
        assert_array_almost_equal(zf_full, zf_sing)

        # lfilter does not prepend ones
        assert_raises(ValueError, lfilter, b, a, x, -1, np.ones(zi_size))

    def test_scalar_a(self):
        # a can be a scalar.
        x = self.generate(6)
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([1])
        y_r = self.convert_dtype([0, 1, 2, 2, 2, 2])

        y = lfilter(b, a[0], x)
        assert_array_almost_equal(y, y_r)

    def test_zi_some_singleton_dims(self):
        # lfilter doesn't really broadcast (no prepending of 1's).  But does
        # do singleton expansion if x and zi have the same ndim.  This was
        # broken only if a subset of the axes were singletons (gh-4681).
        x = self.convert_dtype(np.zeros((3,2,5), 'l'))
        b = self.convert_dtype(np.ones(5, 'l'))
        a = self.convert_dtype(np.array([1,0,0]))
        zi = np.ones((3,1,4), 'l')
        zi[1,:,:] *= 2
        zi[2,:,:] *= 3
        zi = self.convert_dtype(zi)

        zf_expected = self.convert_dtype(np.zeros((3,2,4), 'l'))
        y_expected = np.zeros((3,2,5), 'l')
        y_expected[:,:,:4] = [[[1]], [[2]], [[3]]]
        y_expected = self.convert_dtype(y_expected)

        # IIR
        y_iir, zf_iir = lfilter(b, a, x, -1, zi)
        assert_array_almost_equal(y_iir, y_expected)
        assert_array_almost_equal(zf_iir, zf_expected)

        # FIR
        y_fir, zf_fir = lfilter(b, a[0], x, -1, zi)
        assert_array_almost_equal(y_fir, y_expected)
        assert_array_almost_equal(zf_fir, zf_expected)

    def base_bad_size_zi(self, b, a, x, axis, zi):
        b = self.convert_dtype(b)
        a = self.convert_dtype(a)
        x = self.convert_dtype(x)
        zi = self.convert_dtype(zi)
        assert_raises(ValueError, lfilter, b, a, x, axis, zi)

    def test_bad_size_zi(self):
        # rank 1
        x1 = np.arange(6)
        self.base_bad_size_zi([1], [1], x1, -1, [1])
        self.base_bad_size_zi([1, 1], [1], x1, -1, [0, 1])
        self.base_bad_size_zi([1, 1], [1], x1, -1, [[0]])
        self.base_bad_size_zi([1, 1], [1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1, 1], [1], x1, -1, [[0]])
        self.base_bad_size_zi([1, 1, 1], [1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1], [1, 1], x1, -1, [0, 1])
        self.base_bad_size_zi([1], [1, 1], x1, -1, [[0]])
        self.base_bad_size_zi([1], [1, 1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [[0], [1]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0, 1, 2, 3])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [[0], [1]])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0, 1, 2, 3])

        # rank 2
        x2 = np.arange(12).reshape((4,3))
        # for axis=0 zi.shape should == (max(len(a),len(b))-1, 3)
        self.base_bad_size_zi([1], [1], x2, 0, [0])

        # for each of these there are 5 cases tested (in this order):
        # 1. not deep enough, right # elements
        # 2. too deep, right # elements
        # 3. right depth, right # elements, transposed
        # 4. right depth, too few elements
        # 5. right depth, too many elements

        self.base_bad_size_zi([1, 1], [1], x2, 0, [0,1,2])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[[0,1,2]]])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[0], [1], [2]])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[0,1]])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[0,1,2,3]])

        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [0,1,2,3,4,5])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[[0,1,2],[3,4,5]]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0,1],[2,3]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0,1,2,3],[4,5,6,7]])

        self.base_bad_size_zi([1], [1, 1], x2, 0, [0,1,2])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[[0,1,2]]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[0], [1], [2]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[0,1]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[0,1,2,3]])

        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [0,1,2,3,4,5])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[[0,1,2],[3,4,5]]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0,1],[2,3]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0,1,2,3],[4,5,6,7]])

        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [0,1,2,3,4,5])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[[0,1,2],[3,4,5]]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0,1],[2,3]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0,1,2,3],[4,5,6,7]])

        # for axis=1 zi.shape should == (4, max(len(a),len(b))-1)
        self.base_bad_size_zi([1], [1], x2, 1, [0])

        self.base_bad_size_zi([1, 1], [1], x2, 1, [0,1,2,3])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[[0],[1],[2],[3]]])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[0, 1, 2, 3]])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[0],[1],[2]])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[0],[1],[2],[3],[4]])

        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [0,1,2,3,4,5,6,7])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[[0,1],[2,3],[4,5],[6,7]]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0,1,2,3],[4,5,6,7]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0,1],[2,3],[4,5],[6,7],[8,9]])

        self.base_bad_size_zi([1], [1, 1], x2, 1, [0,1,2,3])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[[0],[1],[2],[3]]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[0, 1, 2, 3]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[0],[1],[2]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[0],[1],[2],[3],[4]])

        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [0,1,2,3,4,5,6,7])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[[0,1],[2,3],[4,5],[6,7]]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0,1,2,3],[4,5,6,7]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0,1],[2,3],[4,5],[6,7],[8,9]])

        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [0,1,2,3,4,5,6,7])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[[0,1],[2,3],[4,5],[6,7]]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0,1,2,3],[4,5,6,7]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0,1],[2,3],[4,5],[6,7],[8,9]])

    def test_empty_zi(self):
        # Regression test for #880: empty array for zi crashes.
        x = self.generate((5,))
        a = self.convert_dtype([1])
        b = self.convert_dtype([1])
        zi = self.convert_dtype([])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, x)
        assert_equal(zf.dtype, self.dtype)
        assert_equal(zf.size, 0)

    # def test_lfiltic_bad_zi(self):
    #     # Regression test for #3699: bad initial conditions
    #     a = self.convert_dtype([1])
    #     b = self.convert_dtype([1])
    #     # "y" sets the datatype of zi, so it truncates if int
    #     zi = lfiltic(b, a, [1., 0])
    #     zi_1 = lfiltic(b, a, [1, 0])
    #     zi_2 = lfiltic(b, a, [True, False])
    #     assert_array_equal(zi, zi_1)
    #     assert_array_equal(zi, zi_2)

    def test_short_x_FIR(self):
        # regression test for #5116
        # x shorter than b, with non None zi fails
        a = self.convert_dtype([1])
        b = self.convert_dtype([1, 0, -1])
        zi = self.convert_dtype([2, 7])
        x = self.convert_dtype([72])
        ye = self.convert_dtype([74])
        zfe = self.convert_dtype([7, -72])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, ye)
        assert_array_almost_equal(zf, zfe)

    def test_short_x_IIR(self):
        # regression test for #5116
        # x shorter than b, with non None zi fails
        a = self.convert_dtype([1, 1])
        b = self.convert_dtype([1, 0, -1])
        zi = self.convert_dtype([2, 7])
        x = self.convert_dtype([72])
        ye = self.convert_dtype([74])
        zfe = self.convert_dtype([-67, -72])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, ye)
        assert_array_almost_equal(zf, zfe)

    def test_do_not_modify_a_b_IIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, -1])
        b0 = b.copy()
        a = self.convert_dtype([0.5, -0.5])
        a0 = a.copy()
        y_r = self.convert_dtype([0, 2, 4, 6, 8, 10.])
        y_f = lfilter(b, a, x)
        assert_array_almost_equal(y_f, y_r)
        assert_equal(b, b0)
        assert_equal(a, a0)

    def test_do_not_modify_a_b_FIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 0, 1])
        b0 = b.copy()
        a = self.convert_dtype([2])
        a0 = a.copy()
        y_r = self.convert_dtype([0, 0.5, 1, 2, 3, 4.])
        y_f = lfilter(b, a, x)
        assert_array_almost_equal(y_f, y_r)
        assert_equal(b, b0)
        assert_equal(a, a0)

class TestLinearFilterFloat32(_TestLinearFilter):
    dtype = np.dtype('f')


class TestLinearFilterFloat64(_TestLinearFilter):
    dtype = np.dtype('d')


class TestLinearFilterFloatExtended(_TestLinearFilter):
    dtype = np.dtype('g')


class TestLinearFilterComplex64(_TestLinearFilter):
    dtype = np.dtype('F')


class TestLinearFilterComplex128(_TestLinearFilter):
    dtype = np.dtype('D')


class TestLinearFilterComplexExtended(_TestLinearFilter):
    dtype = np.dtype('G')

class TestLinearFilterDecimal(_TestLinearFilter):
    dtype = np.dtype('O')

    def type(self, x):
        return Decimal(str(x))


class TestLinearFilterObject(_TestLinearFilter):
    dtype = np.dtype('O')
    type = float


def test_lfilter_bad_object():
    # lfilter: object arrays with non-numeric objects raise TypeError.
    # Regression test for ticket #1452.
    assert_raises(TypeError, lfilter, [1.0], [1.0], [1.0, None, 2.0])
    assert_raises(TypeError, lfilter, [1.0], [None], [1.0, 2.0, 3.0])
    assert_raises(TypeError, lfilter, [None], [1.0], [1.0, 2.0, 3.0])
