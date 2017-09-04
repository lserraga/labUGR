from __future__ import division, print_function, absolute_import

import warnings

from distutils.version import LooseVersion
import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_equal, assert_,
                           assert_allclose, assert_warns)
import pytest
from pytest import raises as assert_raises

from labugr.filters import (freqs, freqz)

class TestFreqs(object):

    def test_basic(self):
        _, h = freqs([1.0], [1.0], worN=8)
        assert_array_almost_equal(h, np.ones(8))

    def test_output(self):
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        w = [0.1, 1, 10, 100]
        num = [1]
        den = [1, 1]
        w, H = freqs(num, den, worN=w)
        s = w * 1j
        expected = 1 / (s + 1)
        assert_array_almost_equal(H.real, expected.real)
        assert_array_almost_equal(H.imag, expected.imag)

    def test_freq_range(self):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # Expected range is from 0.01 to 10.
        num = [1]
        den = [1, 1]
        n = 10
        expected_w = np.logspace(-2, 1, n)
        w, H = freqs(num, den, worN=n)
        assert_array_almost_equal(w, expected_w)

    def test_plot(self):

        def plot(w, h):
            assert_array_almost_equal(h, np.ones(8))

        assert_raises(ZeroDivisionError, freqs, [1.0], [1.0], worN=8,
                      plot=lambda w, h: 1 / 0)
        freqs([1.0], [1.0], worN=8, plot=plot)


class TestFreqz(object):

    def test_ticket1441(self):
        """Regression test for ticket 1441."""
        # Because freqz previously used arange instead of linspace,
        # when N was large, it would return one more point than
        # requested.
        N = 100000
        w, h = freqz([1.0], worN=N)
        assert_equal(w.shape, (N,))

    def test_basic(self):
        w, h = freqz([1.0], worN=8)
        assert_array_almost_equal(w, np.pi * np.arange(8.0) / 8)
        assert_array_almost_equal(h, np.ones(8))

    def test_basic_whole(self):
        w, h = freqz([1.0], worN=8, whole=True)
        assert_array_almost_equal(w, 2 * np.pi * np.arange(8.0) / 8)
        assert_array_almost_equal(h, np.ones(8))

    def test_plot(self):

        def plot(w, h):
            assert_array_almost_equal(w, np.pi * np.arange(8.0) / 8)
            assert_array_almost_equal(h, np.ones(8))

        assert_raises(ZeroDivisionError, freqz, [1.0], worN=8,
                      plot=lambda w, h: 1 / 0)
        freqz([1.0], worN=8, plot=plot)

