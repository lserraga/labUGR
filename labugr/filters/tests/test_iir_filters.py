from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_equal, assert_,
                           assert_allclose, assert_warns)
from pytest import raises as assert_raises

from labugr.filters.iir_filters import (buttap, cheb1ap, cheb2ap, butter,
										buttord)
from labugr.filters.spectral import freqz, freqs

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