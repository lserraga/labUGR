# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from decimal import Decimal

import pytest
from pytest import raises as assert_raises
from numpy.testing import (
    assert_equal,
    assert_almost_equal, assert_array_equal, assert_array_almost_equal,
    assert_allclose, assert_, assert_warns)

from numpy import array, arange
import numpy as np

from labugr.signal import convolve, correlate, fftconvolve, choose_conv_method

class _TestConvolve(object):

    def test_basic(self):
        a = [3, 4, 5, 6, 5, 4]
        b = [1, 2, 3]
        c = convolve(a, b)
        assert_array_equal(c, array([3, 10, 22, 28, 32, 32, 23, 12]))

    def test_same(self):
        a = [3, 4, 5]
        b = [1, 2, 3, 4]
        c = convolve(a, b, mode="same")
        assert_array_equal(c, array([10, 22, 34]))

    def test_same_eq(self):
        a = [3, 4, 5]
        b = [1, 2, 3]
        c = convolve(a, b, mode="same")
        assert_array_equal(c, array([10, 22, 22]))

    def test_complex(self):
        x = array([1 + 1j, 2 + 1j, 3 + 1j])
        y = array([1 + 1j, 2 + 1j])
        z = convolve(x, y)
        assert_array_equal(z, array([2j, 2 + 6j, 5 + 8j, 5 + 5j]))

    def test_zero_rank(self):
        a = 1289
        b = 4567
        c = convolve(a, b)
        assert_equal(c, a * b)

    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        c = convolve(a, b)
        assert_equal(c, a * b)

    def test_2d_arrays(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve(a, b)
        d = array([[2, 7, 16, 17, 12],
                   [10, 30, 62, 58, 38],
                   [12, 31, 58, 49, 30]])
        assert_array_equal(c, d)

    def test_input_swapping(self):
        small = arange(8).reshape(2, 2, 2)
        big = 1j * arange(27).reshape(3, 3, 3)
        big += arange(27)[::-1].reshape(3, 3, 3)

        out_array = array(
            [[[0 + 0j, 26 + 0j, 25 + 1j, 24 + 2j],
              [52 + 0j, 151 + 5j, 145 + 11j, 93 + 11j],
              [46 + 6j, 133 + 23j, 127 + 29j, 81 + 23j],
              [40 + 12j, 98 + 32j, 93 + 37j, 54 + 24j]],

             [[104 + 0j, 247 + 13j, 237 + 23j, 135 + 21j],
              [282 + 30j, 632 + 96j, 604 + 124j, 330 + 86j],
              [246 + 66j, 548 + 180j, 520 + 208j, 282 + 134j],
              [142 + 66j, 307 + 161j, 289 + 179j, 153 + 107j]],

             [[68 + 36j, 157 + 103j, 147 + 113j, 81 + 75j],
              [174 + 138j, 380 + 348j, 352 + 376j, 186 + 230j],
              [138 + 174j, 296 + 432j, 268 + 460j, 138 + 278j],
              [70 + 138j, 145 + 323j, 127 + 341j, 63 + 197j]],

             [[32 + 72j, 68 + 166j, 59 + 175j, 30 + 100j],
              [68 + 192j, 139 + 433j, 117 + 455j, 57 + 255j],
              [38 + 222j, 73 + 499j, 51 + 521j, 21 + 291j],
              [12 + 144j, 20 + 318j, 7 + 331j, 0 + 182j]]])

        assert_array_equal(convolve(small, big, 'full'), out_array)
        assert_array_equal(convolve(big, small, 'full'), out_array)
        assert_array_equal(convolve(small, big, 'same'),
                           out_array[1:3, 1:3, 1:3])
        assert_array_equal(convolve(big, small, 'same'),
                           out_array[0:3, 0:3, 0:3])
        assert_array_equal(convolve(small, big, 'valid'),
                           out_array[1:3, 1:3, 1:3])
        assert_array_equal(convolve(big, small, 'valid'),
                           out_array[1:3, 1:3, 1:3])

class TestConvolve(_TestConvolve):

    def test_valid_mode2(self):
        # See gh-5897
        a = [1, 2, 3, 6, 5, 3]
        b = [2, 3, 4, 5, 3, 4, 2, 2, 1]
        expected = [70, 78, 73, 65]

        out = convolve(a, b, 'valid')
        assert_array_equal(out, expected)

        out = convolve(b, a, 'valid')
        assert_array_equal(out, expected)

        a = [1 + 5j, 2 - 1j, 3 + 0j]
        b = [2 - 3j, 1 + 0j]
        expected = [2 - 3j, 8 - 10j]

        out = convolve(a, b, 'valid')
        assert_array_equal(out, expected)

        out = convolve(b, a, 'valid')
        assert_array_equal(out, expected)

    def test_same_mode(self):
        a = [1, 2, 3, 3, 1, 2]
        b = [1, 4, 3, 4, 5, 6, 7, 4, 3, 2, 1, 1, 3]
        c = convolve(a, b, 'same')
        d = array([57, 61, 63, 57, 45, 36])
        assert_array_equal(c, d)

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))

        assert_raises(ValueError, convolve, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, convolve, *(b, a), **{'mode': 'valid'})

    def test_convolve_method(self, n=100):
        types = sum([t for _, t in np.sctypes.items()], [])
        types = {np.dtype(t).name for t in types}

        # These types include 'bool' and all precisions (int8, float32, etc)
        # The removed types throw errors in correlate or fftconvolve
        for dtype in ['complex256', 'complex192', 'float128', 'float96',
                      'str', 'void', 'bytes', 'object', 'unicode', 'string']:
            if dtype in types:
                types.remove(dtype)

        args = [(t1, t2, mode) for t1 in types for t2 in types
                               for mode in ['valid', 'full', 'same']]

        # These are random arrays, which means test is much stronger than
        # convolving testing by convolving two np.ones arrays
        np.random.seed(42)
        array_types = {'i': np.random.choice([0, 1], size=n),
                       'f': np.random.randn(n)}
        array_types['b'] = array_types['u'] = array_types['i']
        array_types['c'] = array_types['f'] + 0.5j*array_types['f']

        for t1, t2, mode in args:
            x1 = array_types[np.dtype(t1).kind].astype(t1)
            x2 = array_types[np.dtype(t2).kind].astype(t2)

            results = {key: convolve(x1, x2, method=key, mode=mode)
                       for key in ['fft', 'direct']}

            assert_equal(results['fft'].dtype, results['direct'].dtype)

            if 'bool' in t1 and 'bool' in t2:
                assert_equal(choose_conv_method(x1, x2), 'direct')
                continue

            # Found by experiment. Found approx smallest value for (rtol, atol)
            # threshold to have tests pass.
            if any([t in {'complex64', 'float32'} for t in [t1, t2]]):
                kwargs = {'rtol': 1.0e-4, 'atol': 1e-6}
            elif 'float16' in [t1, t2]:
                # atol is default for np.allclose
                kwargs = {'rtol': 1e-3, 'atol': 1e-8}
            else:
                # defaults for np.allclose (different from assert_allclose)
                kwargs = {'rtol': 1e-5, 'atol': 1e-8}

            assert_allclose(results['fft'], results['direct'], **kwargs)

    def test_convolve_method_large_input(self):
        # This is really a test that convolving two large integers goes to the
        # direct method even if they're in the fft method.
        for n in [10, 20, 50, 51, 52, 53, 54, 60, 62]:
            z = np.array([2**n], dtype=np.int64)
            fft = convolve(z, z, method='fft')
            direct = convolve(z, z, method='direct')

            # this is the case when integer precision gets to us
            # issue #6076 has more detail, hopefully more tests after resolved
            if n < 50:
                assert_equal(fft, direct)
                assert_equal(fft, 2**(2*n))
                assert_equal(direct, 2**(2*n))

class TestFFTConvolve(object):

    def test_real(self):
        x = array([1, 2, 3])
        assert_array_almost_equal(fftconvolve(x, x), [1, 4, 10, 12, 9.])

    def test_complex(self):
        x = array([1 + 1j, 2 + 2j, 3 + 3j])
        assert_array_almost_equal(fftconvolve(x, x),
                                  [0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])

    def test_2d_real_same(self):
        a = array([[1, 2, 3], [4, 5, 6]])
        assert_array_almost_equal(fftconvolve(a, a),
                                  array([[1, 4, 10, 12, 9],
                                         [8, 26, 56, 54, 36],
                                         [16, 40, 73, 60, 36]]))

    def test_2d_complex_same(self):
        a = array([[1 + 2j, 3 + 4j, 5 + 6j], [2 + 1j, 4 + 3j, 6 + 5j]])
        c = fftconvolve(a, a)
        d = array([[-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j],
                   [10j, 44j, 118j, 156j, 122j],
                   [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]])
        assert_array_almost_equal(c, d)

    def test_real_same_mode(self):
        a = array([1, 2, 3])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        c = fftconvolve(a, b, 'same')
        d = array([35., 41., 47.])
        assert_array_almost_equal(c, d)

    def test_real_same_mode2(self):
        a = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        b = array([1, 2, 3])
        c = fftconvolve(a, b, 'same')
        d = array([9., 20., 25., 35., 41., 47., 39., 28., 2.])
        assert_array_almost_equal(c, d)

    def test_valid_mode(self):
        # See gh-5897
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24., 31., 41., 43., 49., 25., 12.])

        out = fftconvolve(a, b, 'valid')
        assert_array_almost_equal(out, expected)

        out = fftconvolve(b, a, 'valid')
        assert_array_almost_equal(out, expected)

        a = array([3 - 1j, 2 + 7j, 1 + 0j])
        b = array([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        expected = array([45. + 12.j, 30. + 23.j, 48 + 32.j])

        out = fftconvolve(a, b, 'valid')
        assert_array_almost_equal(out, expected)

        out = fftconvolve(b, a, 'valid')
        assert_array_almost_equal(out, expected)

    def test_real_valid_mode(self):
        a = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        b = array([3, 2, 1])
        d = array([24., 31., 41., 43., 49., 25., 12.])

        c = fftconvolve(a, b, 'valid')
        assert_array_almost_equal(c, d)

        # See gh-5897
        c = fftconvolve(b, a, 'valid')
        assert_array_almost_equal(c, d)

    def test_empty(self):
        # Regression test for #1745: crashes with 0-length input.
        assert_(fftconvolve([], []).size == 0)
        assert_(fftconvolve([5, 6], []).size == 0)
        assert_(fftconvolve([], [7]).size == 0)

    def test_zero_rank(self):
        a = array(4967)
        b = array(3920)
        c = fftconvolve(a, b)
        assert_equal(c, a * b)

    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        c = fftconvolve(a, b)
        assert_equal(c, a * b)

    def test_random_data(self):
        np.random.seed(1234)
        a = np.random.rand(1233) + 1j * np.random.rand(1233)
        b = np.random.rand(1321) + 1j * np.random.rand(1321)
        c = fftconvolve(a, b, 'full')
        d = np.convolve(a, b, 'full')
        assert_(np.allclose(c, d, rtol=1e-10))

    # @pytest.mark.slow
    # def test_many_sizes(self):
    #     np.random.seed(1234)

    #     def ns():
    #         for j in range(1, 100):
    #             yield j
    #         for j in range(1000, 1500):
    #             yield j
    #         for k in range(50):
    #             yield np.random.randint(1001, 10000)

    #     for n in ns():
    #         msg = 'n=%d' % (n,)
    #         a = np.random.rand(n) + 1j * np.random.rand(n)
    #         b = np.random.rand(n) + 1j * np.random.rand(n)
    #         c = fftconvolve(a, b, 'full')
    #         d = np.convolve(a, b, 'full')
    #         assert_allclose(c, d, atol=1e-10, err_msg=msg)

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))

        assert_raises(ValueError, fftconvolve, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, fftconvolve, *(b, a), **{'mode': 'valid'})




@pytest.mark.parametrize('dt', [np.ubyte, np.byte, np.ushort, np.short, np.uint, int,
                 np.ulonglong, np.ulonglong, np.float32, np.float64,
                 np.longdouble, Decimal])
class TestCorrelateReal(object):
    def _setup_rank1(self, dt):
        a = np.linspace(0, 3, 4).astype(dt)
        b = np.linspace(1, 2, 2).astype(dt)

        y_r = np.array([0, 2, 5, 8, 3]).astype(dt)
        return a, b, y_r

    def test_method(self, dt):
        if dt == Decimal:
            method = choose_conv_method([Decimal(4)], [Decimal(3)])
            assert_equal(method, 'direct')
        else:
            a, b, y_r = self._setup_rank3(dt)
            y_fft = correlate(a, b, method='fft')
            y_direct = correlate(a, b, method='direct')

            assert_array_almost_equal(y_r, y_fft)
            assert_array_almost_equal(y_r, y_direct)
            assert_equal(y_fft.dtype, dt)
            assert_equal(y_direct.dtype, dt)

    def test_rank1_valid(self, dt):
        a, b, y_r = self._setup_rank1(dt)
        y = correlate(a, b, 'valid')
        assert_array_almost_equal(y, y_r[1:4])
        assert_equal(y.dtype, dt)

        # See gh-5897
        y = correlate(b, a, 'valid')
        assert_array_almost_equal(y, y_r[1:4][::-1])
        assert_equal(y.dtype, dt)

    def test_rank1_same(self, dt):
        a, b, y_r = self._setup_rank1(dt)
        y = correlate(a, b, 'same')
        assert_array_almost_equal(y, y_r[:-1])
        assert_equal(y.dtype, dt)

    def test_rank1_full(self, dt):
        a, b, y_r = self._setup_rank1(dt)
        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r)
        assert_equal(y.dtype, dt)

    def _setup_rank3(self, dt):
        a = np.linspace(0, 39, 40).reshape((2, 4, 5), order='F').astype(
            dt)
        b = np.linspace(0, 23, 24).reshape((2, 3, 4), order='F').astype(
            dt)

        y_r = array([[[0., 184., 504., 912., 1360., 888., 472., 160.],
                      [46., 432., 1062., 1840., 2672., 1698., 864., 266.],
                      [134., 736., 1662., 2768., 3920., 2418., 1168., 314.],
                      [260., 952., 1932., 3056., 4208., 2580., 1240., 332.],
                      [202., 664., 1290., 1984., 2688., 1590., 712., 150.],
                      [114., 344., 642., 960., 1280., 726., 296., 38.]],

                     [[23., 400., 1035., 1832., 2696., 1737., 904., 293.],
                      [134., 920., 2166., 3680., 5280., 3306., 1640., 474.],
                      [325., 1544., 3369., 5512., 7720., 4683., 2192., 535.],
                      [571., 1964., 3891., 6064., 8272., 4989., 2324., 565.],
                      [434., 1360., 2586., 3920., 5264., 3054., 1312., 230.],
                      [241., 700., 1281., 1888., 2496., 1383., 532., 39.]],

                     [[22., 214., 528., 916., 1332., 846., 430., 132.],
                      [86., 484., 1098., 1832., 2600., 1602., 772., 206.],
                      [188., 802., 1698., 2732., 3788., 2256., 1018., 218.],
                      [308., 1006., 1950., 2996., 4052., 2400., 1078., 230.],
                      [230., 692., 1290., 1928., 2568., 1458., 596., 78.],
                      [126., 354., 636., 924., 1212., 654., 234., 0.]]],
                    dtype=dt)

        return a, b, y_r

    def test_rank3_valid(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y = correlate(a, b, "valid")
        assert_array_almost_equal(y, y_r[1:2, 2:4, 3:5])
        assert_equal(y.dtype, dt)

        # See gh-5897
        y = correlate(b, a, "valid")
        assert_array_almost_equal(y, y_r[1:2, 2:4, 3:5][::-1, ::-1, ::-1])
        assert_equal(y.dtype, dt)

    def test_rank3_same(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y = correlate(a, b, "same")
        assert_array_almost_equal(y, y_r[0:-1, 1:-1, 1:-2])
        assert_equal(y.dtype, dt)

    def test_rank3_all(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y = correlate(a, b)
        assert_array_almost_equal(y, y_r)
        assert_equal(y.dtype, dt)

    def test_invalid_shapes(self, dt):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))

        assert_raises(ValueError, correlate, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, correlate, *(b, a), **{'mode': 'valid'})


@pytest.mark.parametrize('dt', [np.csingle, np.cdouble, np.clongdouble])
class TestCorrelateComplex(object):
    # The decimal precision to be used for comparing results.
    # This value will be passed as the 'decimal' keyword argument of
    # assert_array_almost_equal().

    def decimal(self, dt):
        return int(2 * np.finfo(dt).precision / 3)

    def _setup_rank1(self, dt, mode):
        np.random.seed(9)
        a = np.random.randn(10).astype(dt)
        a += 1j * np.random.randn(10).astype(dt)
        b = np.random.randn(8).astype(dt)
        b += 1j * np.random.randn(8).astype(dt)

        y_r = (correlate(a.real, b.real, mode=mode) +
               correlate(a.imag, b.imag, mode=mode)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag, mode=mode) +
                     correlate(a.imag, b.real, mode=mode))
        return a, b, y_r

    def test_rank1_valid(self, dt):
        a, b, y_r = self._setup_rank1(dt, 'valid')
        y = correlate(a, b, 'valid')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

        # See gh-5897
        y = correlate(b, a, 'valid')
        assert_array_almost_equal(y, y_r[::-1].conj(), decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_same(self, dt):
        a, b, y_r = self._setup_rank1(dt, 'same')
        y = correlate(a, b, 'same')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_full(self, dt):
        a, b, y_r = self._setup_rank1(dt, 'full')
        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_swap_full(self, dt):
        d = np.array([0.+0.j, 1.+1.j, 2.+2.j], dtype=dt)
        k = np.array([1.+3.j, 2.+4.j, 3.+5.j, 4.+6.j], dtype=dt)
        y = correlate(d, k)
        assert_equal(y, [0.+0.j, 10.-2.j, 28.-6.j, 22.-6.j, 16.-6.j, 8.-4.j])

    def test_swap_same(self, dt):
        d = [0.+0.j, 1.+1.j, 2.+2.j]
        k = [1.+3.j, 2.+4.j, 3.+5.j, 4.+6.j]
        y = correlate(d, k, mode="same")
        assert_equal(y, [10.-2.j, 28.-6.j, 22.-6.j])

    def test_rank3(self, dt):
        a = np.random.randn(10, 8, 6).astype(dt)
        a += 1j * np.random.randn(10, 8, 6).astype(dt)
        b = np.random.randn(8, 6, 4).astype(dt)
        b += 1j * np.random.randn(8, 6, 4).astype(dt)

        y_r = (correlate(a.real, b.real)
               + correlate(a.imag, b.imag)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag) + correlate(a.imag, b.real))

        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
        assert_equal(y.dtype, dt)
