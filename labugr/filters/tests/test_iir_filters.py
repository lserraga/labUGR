from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_equal, assert_,
                           assert_allclose, assert_warns)
from pytest import raises as assert_raises

from labugr.filters.iir_filters import (buttap, cheb1ap, cheb2ap, butter,
										cheby1, cheby2, buttord, cheb1ord,
										cheb2ord)
from labugr.filters.spectral import freqz, freqs


class TestButtord(object):

    def test_lowpass(self):
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        N, Wn = buttord(wp, ws, rp, rs, False)
        b, a = butter(N, Wn, 'lowpass', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs)

        assert_equal(N, 16)
        assert_allclose(Wn, 2.0002776782743284e-01, rtol=1e-15)

    def test_highpass(self):
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        N, Wn = buttord(wp, ws, rp, rs, False)
        b, a = butter(N, Wn, 'highpass', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs)

        assert_equal(N, 18)
        assert_allclose(Wn, 2.9996603079132672e-01, rtol=1e-15)

    def test_bandpass(self):
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        N, Wn = buttord(wp, ws, rp, rs, False)
        b, a = butter(N, Wn, 'bandpass', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)

        assert_equal(N, 18)
        assert_allclose(Wn, [1.9998742411409134e-01, 5.0002139595676276e-01],
                        rtol=1e-15)

    def test_bandstop(self):
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        N, Wn = buttord(wp, ws, rp, rs, False)
        b, a = butter(N, Wn, 'bandstop', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp,
                          dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]),
                          -rs)

        assert_equal(N, 20)
        assert_allclose(Wn, [1.4759432329294042e-01, 5.9997365985276407e-01],
                        rtol=1e-6)

    def test_analog(self):
        wp = 200
        ws = 600
        rp = 3
        rs = 60
        N, Wn = buttord(wp, ws, rp, rs, True)
        b, a = butter(N, Wn, 'lowpass', True)
        w, h = freqs(b, a)
        assert_array_less(-rp, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs)

        assert_equal(N, 7)
        assert_allclose(Wn, 2.0006785355671877e+02, rtol=1e-15)

        n, Wn = buttord(1, 550/450, 1, 26, analog=True)
        assert_equal(n, 19)
        assert_allclose(Wn, 1.0361980524629517, rtol=1e-15)

        assert_equal(buttord(1, 1.2, 1, 80, analog=True)[0], 55)

class TestCheb1ord(object):

    def test_lowpass(self):
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        N, Wn = cheb1ord(wp, ws, rp, rs, False)
        b, a = cheby1(N, rp, Wn, 'low', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs + 0.1)

        assert_equal(N, 8)
        assert_allclose(Wn, 0.2, rtol=1e-15)

    def test_highpass(self):
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        N, Wn = cheb1ord(wp, ws, rp, rs, False)
        b, a = cheby1(N, rp, Wn, 'high', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)

        assert_equal(N, 9)
        assert_allclose(Wn, 0.3, rtol=1e-15)

    def test_bandpass(self):
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        N, Wn = cheb1ord(wp, ws, rp, rs, False)
        b, a = cheby1(N, rp, Wn, 'band', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)

        assert_equal(N, 9)
        assert_allclose(Wn, [0.2, 0.5], rtol=1e-15)

    def test_bandstop(self):
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        N, Wn = cheb1ord(wp, ws, rp, rs, False)
        b, a = cheby1(N, rp, Wn, 'stop', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]),
                          -rs + 0.1)

        assert_equal(N, 10)
        assert_allclose(Wn, [0.14758232569947785, 0.6], rtol=1e-5)

    def test_analog(self):
        wp = 700
        ws = 100
        rp = 3
        rs = 70
        N, Wn = cheb1ord(wp, ws, rp, rs, True)
        b, a = cheby1(N, rp, Wn, 'high', True)
        w, h = freqs(b, a)
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)

        assert_equal(N, 4)
        assert_allclose(Wn, 700, rtol=1e-15)

        assert_equal(cheb1ord(1, 1.2, 1, 80, analog=True)[0], 17)


class TestCheb2ord(object):

    def test_lowpass(self):
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        N, Wn = cheb2ord(wp, ws, rp, rs, False)
        b, a = cheby2(N, rs, Wn, 'lp', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs + 0.1)

        assert_equal(N, 8)
        assert_allclose(Wn, 0.28647639976553163, rtol=1e-15)

    def test_highpass(self):
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        N, Wn = cheb2ord(wp, ws, rp, rs, False)
        b, a = cheby2(N, rs, Wn, 'hp', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)

        assert_equal(N, 9)
        assert_allclose(Wn, 0.20697492182903282, rtol=1e-15)

    def test_bandpass(self):
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        N, Wn = cheb2ord(wp, ws, rp, rs, False)
        b, a = cheby2(N, rs, Wn, 'bp', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)

        assert_equal(N, 9)
        assert_allclose(Wn, [0.14876937565923479, 0.59748447842351482],
                        rtol=1e-15)

    def test_bandstop(self):
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        N, Wn = cheb2ord(wp, ws, rp, rs, False)
        b, a = cheby2(N, rs, Wn, 'bs', False)
        w, h = freqz(b, a)
        w /= np.pi
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]),
                          -rs + 0.1)

        assert_equal(N, 10)
        assert_allclose(Wn, [0.19926249974781743, 0.50125246585567362],
                        rtol=1e-6)

    def test_analog(self):
        wp = [20, 50]
        ws = [10, 60]
        rp = 3
        rs = 80
        N, Wn = cheb2ord(wp, ws, rp, rs, True)
        b, a = cheby2(N, rs, Wn, 'bp', True)
        w, h = freqs(b, a)
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)

        assert_equal(N, 11)
        assert_allclose(Wn, [1.673740595370124e+01, 5.974641487254268e+01],
                        rtol=1e-15)

class TestPrototypeType(object):

    def test_output_type(self):
        # Prototypes should consistently output arrays, not lists
        # https://github.com/scipy/scipy/pull/441
        for func in (buttap,
                     lambda N: cheb1ap(N, 1),
                     lambda N: cheb2ap(N, 20)):
            for N in range(7):
                z, p, k = func(N)
                assert_(isinstance(z, np.ndarray))
                assert_(isinstance(p, np.ndarray))

def dB(x):
    # Return magnitude in decibels
    return 20 * np.log10(abs(x))
