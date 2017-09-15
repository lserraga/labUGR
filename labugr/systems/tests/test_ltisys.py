from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
                           assert_)
from pytest import raises as assert_raises

from labugr.integrate._lib._numpy_compat import suppress_warnings
from labugr.systems.ltisys import (lti, dlsim,
                          dlti, bode, freqresp, lsim, impulse, step,
                          place_poles,
                          TransferFunction, StateSpace, ZerosPolesGain)
from labugr.filters.filters import BadCoefficients


def _assert_poles_close(P1,P2, rtol=1e-8, atol=1e-8):
    """
    Check each pole in P1 is close to a pole in P2 with a 1e-8
    relative tolerance or 1e-8 absolute tolerance (useful for zero poles).
    These tolerances are very strict but the systems tested are known to
    accept these poles so we should not be far from what is requested.
    """
    P2 = P2.copy()
    for p1 in P1:
        found = False
        for p2_idx in range(P2.shape[0]):
            if np.allclose([np.real(p1), np.imag(p1)],
                           [np.real(P2[p2_idx]), np.imag(P2[p2_idx])],
                           rtol, atol):
                found = True
                np.delete(P2, p2_idx)
                break
        if not found:
            raise ValueError("Can't find pole " + str(p1) + " in " + str(P2))


class TestLsim(object):
    def lti_nowarn(self, *args):
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            system = lti(*args)
        return system

    def test_first_order(self):
        # y' = -y
        # exact solution is y(t) = exp(-t)
        system = self.lti_nowarn(-1.,1.,1.,0.)
        t = np.linspace(0,5)
        u = np.zeros_like(t)
        tout, y, x = lsim(system, u, t, X0=[1.0])
        expected_x = np.exp(-tout)
        assert_almost_equal(x, expected_x)
        assert_almost_equal(y, expected_x)

    def test_integrator(self):
        # integrator: y' = u
        system = self.lti_nowarn(0., 1., 1., 0.)
        t = np.linspace(0,5)
        u = t
        tout, y, x = lsim(system, u, t)
        expected_x = 0.5 * tout**2
        assert_almost_equal(x, expected_x)
        assert_almost_equal(y, expected_x)

    def test_double_integrator(self):
        # double integrator: y'' = 2u
        A = np.mat("0. 1.; 0. 0.")
        B = np.mat("0.; 1.")
        C = np.mat("2. 0.")
        system = self.lti_nowarn(A, B, C, 0.)
        t = np.linspace(0,5)
        u = np.ones_like(t)
        tout, y, x = lsim(system, u, t)
        expected_x = np.transpose(np.array([0.5 * tout**2, tout]))
        expected_y = tout**2
        assert_almost_equal(x, expected_x)
        assert_almost_equal(y, expected_y)

    def test_jordan_block(self):
        # Non-diagonalizable A matrix
        #   x1' + x1 = x2
        #   x2' + x2 = u
        #   y = x1
        # Exact solution with u = 0 is y(t) = t exp(-t)
        A = np.mat("-1. 1.; 0. -1.")
        B = np.mat("0.; 1.")
        C = np.mat("1. 0.")
        system = self.lti_nowarn(A, B, C, 0.)
        t = np.linspace(0,5)
        u = np.zeros_like(t)
        tout, y, x = lsim(system, u, t, X0=[0.0, 1.0])
        expected_y = tout * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_miso(self):
        # A system with two state variables, two inputs, and one output.
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = np.array([1.0, 0.0])
        D = np.zeros((1,2))
        system = self.lti_nowarn(A, B, C, D)

        t = np.linspace(0, 5.0, 101)
        u = np.zeros_like(t)
        tout, y, x = lsim(system, u, t, X0=[1.0, 1.0])
        expected_y = np.exp(-tout)
        expected_x0 = np.exp(-tout)
        expected_x1 = np.exp(-2.0*tout)
        assert_almost_equal(y, expected_y)
        assert_almost_equal(x[:,0], expected_x0)
        assert_almost_equal(x[:,1], expected_x1)

    def test_nonzero_initial_time(self):
        system = self.lti_nowarn(-1.,1.,1.,0.)
        t = np.linspace(1,2)
        u = np.zeros_like(t)
        tout, y, x = lsim(system, u, t, X0=[1.0])
        expected_y = np.exp(-tout)
        assert_almost_equal(y, expected_y)


class _TestImpulseFuncs(object):
    # Common tests for impulse/impulse2 (= self.func)

    def test_01(self):
        # First order system: x'(t) + x(t) = u(t)
        # Exact impulse response is x(t) = exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system)
        expected_y = np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_02(self):
        # Specify the desired time values for the output.

        # First order system: x'(t) + x(t) = u(t)
        # Exact impulse response is x(t) = exp(-t).
        system = ([1.0], [1.0,1.0])
        n = 21
        t = np.linspace(0, 2.0, n)
        tout, y = self.func(system, T=t)
        assert_equal(tout.shape, (n,))
        assert_almost_equal(tout, t)
        expected_y = np.exp(-t)
        assert_almost_equal(y, expected_y)

    def test_03(self):
        # Specify an initial condition as a scalar.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact impulse response is x(t) = 4*exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system, X0=3.0)
        expected_y = 4.0 * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_04(self):
        # Specify an initial condition as a list.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact impulse response is x(t) = 4*exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system, X0=[3.0])
        expected_y = 4.0 * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_05(self):
        # Simple integrator: x'(t) = u(t)
        system = ([1.0], [1.0,0.0])
        tout, y = self.func(system)
        expected_y = np.ones_like(tout)
        assert_almost_equal(y, expected_y)

    def test_06(self):
        # Second order system with a repeated root:
        #     x''(t) + 2*x(t) + x(t) = u(t)
        # The exact impulse response is t*exp(-t).
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y = self.func(system)
        expected_y = tout * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_array_like(self):
        # Test that function can accept sequences, scalars.
        system = ([1.0], [1.0, 2.0, 1.0])
        # TODO: add meaningful test where X0 is a list
        tout, y = self.func(system, X0=[3], T=[5, 6])
        tout, y = self.func(system, X0=[3], T=[5])

    def test_array_like2(self):
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y = self.func(system, X0=3, T=5)


class TestImpulse(_TestImpulseFuncs):
    def setup_method(self):
        self.func = impulse


class _TestStepFuncs(object):
    def test_01(self):
        # First order system: x'(t) + x(t) = u(t)
        # Exact step response is x(t) = 1 - exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system)
        expected_y = 1.0 - np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_02(self):
        # Specify the desired time values for the output.

        # First order system: x'(t) + x(t) = u(t)
        # Exact step response is x(t) = 1 - exp(-t).
        system = ([1.0], [1.0,1.0])
        n = 21
        t = np.linspace(0, 2.0, n)
        tout, y = self.func(system, T=t)
        assert_equal(tout.shape, (n,))
        assert_almost_equal(tout, t)
        expected_y = 1 - np.exp(-t)
        assert_almost_equal(y, expected_y)

    def test_03(self):
        # Specify an initial condition as a scalar.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact step response is x(t) = 1 + 2*exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system, X0=3.0)
        expected_y = 1 + 2.0*np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_04(self):
        # Specify an initial condition as a list.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact step response is x(t) = 1 + 2*exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system, X0=[3.0])
        expected_y = 1 + 2.0*np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_05(self):
        # Simple integrator: x'(t) = u(t)
        # Exact step response is x(t) = t.
        system = ([1.0],[1.0,0.0])
        tout, y = self.func(system)
        expected_y = tout
        assert_almost_equal(y, expected_y)

    def test_06(self):
        # Second order system with a repeated root:
        #     x''(t) + 2*x(t) + x(t) = u(t)
        # The exact step response is 1 - (1 + t)*exp(-t).
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y = self.func(system)
        expected_y = 1 - (1 + tout) * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_array_like(self):
        # Test that function can accept sequences, scalars.
        system = ([1.0], [1.0, 2.0, 1.0])
        # TODO: add meaningful test where X0 is a list
        tout, y = self.func(system, T=[5, 6])


class TestStep(_TestStepFuncs):
    def setup_method(self):
        self.func = step

    def test_complex_input(self):
        # Test that complex input doesn't raise an error.
        # `step` doesn't seem to have been designed for complex input, but this
        # works and may be used, so add regression test.  See gh-2654.
        step(([], [-1], 1+0j))


class TestLti(object):
    def test_lti_instantiation(self):
        # Test that lti can be instantiated with sequences, scalars.
        # See PR-225.

        # TransferFunction
        s = lti([1], [-1])
        assert_(isinstance(s, TransferFunction))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)

        # ZerosPolesGain
        s = lti(np.array([]), np.array([-1]), 1)
        assert_(isinstance(s, ZerosPolesGain))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)

        # StateSpace
        s = lti([], [-1], 1)
        s = lti([1], [-1], 1, 3)
        assert_(isinstance(s, StateSpace))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)


class TestStateSpace(object):
    def test_initialization(self):
        # Check that all initializations work
        s = StateSpace(1, 1, 1, 1)
        s = StateSpace([1], [2], [3], [4])
        s = StateSpace(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]),
                       np.array([[1, 0]]), np.array([[0]]))

    def test_conversion(self):
        # Check the conversion functions
        s = StateSpace(1, 2, 3, 4)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        assert_(StateSpace(s) is not s)
        assert_(s.to_ss() is not s)

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_tf() and to_zpk()

        # Getters
        s = StateSpace(1, 1, 1, 1)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])
        assert_(s.dt is None)


class TestTransferFunction(object):
    def test_initialization(self):
        # Check that all initializations work
        s = TransferFunction(1, 1)
        s = TransferFunction([1], [2])
        s = TransferFunction(np.array([1]), np.array([2]))

    def test_conversion(self):
        # Check the conversion functions
        s = TransferFunction([1, 0], [1, -1])
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        assert_(TransferFunction(s) is not s)
        assert_(s.to_tf() is not s)

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_ss() and to_zpk()

        # Getters
        s = TransferFunction([1, 0], [1, -1])
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])


class TestZerosPolesGain(object):
    def test_initialization(self):
        # Check that all initializations work
        s = ZerosPolesGain(1, 1, 1)
        s = ZerosPolesGain([1], [2], 1)
        s = ZerosPolesGain(np.array([1]), np.array([2]), 1)

    def test_conversion(self):
        #Check the conversion functions
        s = ZerosPolesGain(1, 2, 3)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        assert_(ZerosPolesGain(s) is not s)
        assert_(s.to_zpk() is not s)


class Test_bode(object):

    def test_01(self):
        # Test bode() magnitude calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        # cutoff: 1 rad/s, slope: -20 dB/decade
        #   H(s=0.1) ~= 0 dB
        #   H(s=1) ~= -3 dB
        #   H(s=10) ~= -20 dB
        #   H(s=100) ~= -40 dB
        system = lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, mag, phase = bode(system, w=w)
        expected_mag = [0, -3, -20, -40]
        assert_almost_equal(mag, expected_mag, decimal=1)

    def test_02(self):
        # Test bode() phase calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        #   angle(H(s=0.1)) ~= -5.7 deg
        #   angle(H(s=1)) ~= -45 deg
        #   angle(H(s=10)) ~= -84.3 deg
        system = lti([1], [1, 1])
        w = [0.1, 1, 10]
        w, mag, phase = bode(system, w=w)
        expected_phase = [-5.7, -45, -84.3]
        assert_almost_equal(phase, expected_phase, decimal=1)

    def test_03(self):
        # Test bode() magnitude calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, mag, phase = bode(system, w=w)
        jw = w * 1j
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
        expected_mag = 20.0 * np.log10(abs(y))
        assert_almost_equal(mag, expected_mag)

    def test_04(self):
        # Test bode() phase calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, mag, phase = bode(system, w=w)
        jw = w * 1j
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
        expected_phase = np.arctan2(y.imag, y.real) * 180.0 / np.pi
        assert_almost_equal(phase, expected_phase)

    def test_05(self):
        # Test that bode() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        n = 10
        # Expected range is from 0.01 to 10.
        expected_w = np.logspace(-2, 1, n)
        w, mag, phase = bode(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_06(self):
        # Test that bode() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = lti([1], [1, 0])
        w, mag, phase = bode(system, n=2)
        assert_equal(w[0], 0.01)  # a fail would give not-a-number

    def test_07(self):
        # bode() should not fail on a system with pure imaginary poles.
        # The test passes if bode doesn't raise an exception.
        system = lti([1], [1, 0, 100])
        w, mag, phase = bode(system, n=2)

    def test_08(self):
        # Test that bode() return continuous phase, issues/2331.
        system = lti([], [-10, -30, -40, -60, -70], 1)
        w, mag, phase = system.bode(w=np.logspace(-3, 40, 100))
        assert_almost_equal(min(phase), -450, decimal=15)

    def test_from_state_space(self):
        # Ensure that bode works with a system that was created from the
        # state space representation matrices A, B, C, D.  In this case,
        # system.num will be a 2-D array with shape (1, n+1), where (n,n)
        # is the shape of A.
        # A Butterworth lowpass filter is used, so we know the exact
        # frequency response.
        a = np.array([1.0, 2.0, 2.0, 1.0])
        A = companion(a).T
        B = np.array([[0.0], [0.0], [1.0]])
        C = np.array([[1.0, 0.0, 0.0]])
        D = np.array([[0.0]])
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            system = lti(A, B, C, D)
            w, mag, phase = bode(system, n=100)

        expected_magnitude = 20 * np.log10(np.sqrt(1.0 / (1.0 + w**6)))
        assert_almost_equal(mag, expected_magnitude)


class Test_freqresp(object):

    def test_output_manual(self):
        # Test freqresp() output calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        #   re(H(s=0.1)) ~= 0.99
        #   re(H(s=1)) ~= 0.5
        #   re(H(s=10)) ~= 0.0099
        system = lti([1], [1, 1])
        w = [0.1, 1, 10]
        w, H = freqresp(system, w=w)
        expected_re = [0.99, 0.5, 0.0099]
        expected_im = [-0.099, -0.5, -0.099]
        assert_almost_equal(H.real, expected_re, decimal=1)
        assert_almost_equal(H.imag, expected_im, decimal=1)

    def test_output(self):
        # Test freqresp() output calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, H = freqresp(system, w=w)
        s = w * 1j
        expected = np.polyval(system.num, s) / np.polyval(system.den, s)
        assert_almost_equal(H.real, expected.real)
        assert_almost_equal(H.imag, expected.imag)

    def test_freq_range(self):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # Expected range is from 0.01 to 10.
        system = lti([1], [1, 1])
        n = 10
        expected_w = np.logspace(-2, 1, n)
        w, H = freqresp(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_pole_zero(self):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = lti([1], [1, 0])
        w, H = freqresp(system, n=2)
        assert_equal(w[0], 0.01)  # a fail would give not-a-number

    def test_from_state_space(self):
        # Ensure that freqresp works with a system that was created from the
        # state space representation matrices A, B, C, D.  In this case,
        # system.num will be a 2-D array with shape (1, n+1), where (n,n) is
        # the shape of A.
        # A Butterworth lowpass filter is used, so we know the exact
        # frequency response.
        a = np.array([1.0, 2.0, 2.0, 1.0])
        A = companion(a).T
        B = np.array([[0.0],[0.0],[1.0]])
        C = np.array([[1.0, 0.0, 0.0]])
        D = np.array([[0.0]])
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            system = lti(A, B, C, D)
            w, H = freqresp(system, n=100)
        s = w * 1j
        expected = (1.0 / (1.0 + 2*s + 2*s**2 + s**3))
        assert_almost_equal(H.real, expected.real)
        assert_almost_equal(H.imag, expected.imag)

    def test_from_zpk(self):
        # 4th order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([],[-1]*4,[1])
        w = [0.1, 1, 10, 100]
        w, H = freqresp(system, w=w)
        s = w * 1j
        expected = 1 / (s + 1)**4
        assert_almost_equal(H.real, expected.real)
        assert_almost_equal(H.imag, expected.imag)
def companion(a):
    """
    Create a companion matrix.

    Create the companion matrix [1]_ associated with the polynomial whose
    coefficients are given in `a`.

    Parameters
    ----------
    a : (N,) array_like
        1-D array of polynomial coefficients.  The length of `a` must be
        at least two, and ``a[0]`` must not be zero.

    Returns
    -------
    c : (N-1, N-1) ndarray
        The first row of `c` is ``-a[1:]/a[0]``, and the first
        sub-diagonal is all ones.  The data-type of the array is the same
        as the data-type of ``1.0*a[0]``.

    Raises
    ------
    ValueError
        If any of the following are true: a) ``a.ndim != 1``;
        b) ``a.size < 2``; c) ``a[0] == 0``.

    Notes
    -----
    .. versionadded:: 0.8.0

    References
    ----------
    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
        Cambridge University Press, 1999, pp. 146-7.

    Examples
    --------
    >>> from .linalg import companion
    >>> companion([1, -10, 31, -30])
    array([[ 10., -31.,  30.],
           [  1.,   0.,   0.],
           [  0.,   1.,   0.]])

    """
    a = np.atleast_1d(a)

    if a.ndim != 1:
        raise ValueError("Incorrect shape for `a`.  `a` must be "
                         "one-dimensional.")

    if a.size < 2:
        raise ValueError("The length of `a` must be at least 2.")

    if a[0] == 0:
        raise ValueError("The first coefficient in `a` must not be zero.")

    first_row = -a[1:] / (1.0 * a[0])
    n = a.size
    c = np.zeros((n - 1, n - 1), dtype=first_row.dtype)
    c[0] = first_row
    c[list(range(1, n - 1)), list(range(0, n - 2))] = 1
    return c

class TestPlacePoles(object):

    def _check(self, A, B, P, **kwargs):
        """
        Perform the most common tests on the poles computed by place_poles
        and return the Bunch object for further specific tests
        """
        fsf = place_poles(A, B, P, **kwargs)
        expected, _ = np.linalg.eig(A - np.dot(B, fsf.gain_matrix))
        _assert_poles_close(expected,fsf.requested_poles)
        _assert_poles_close(expected,fsf.computed_poles)
        _assert_poles_close(P,fsf.requested_poles)
        return fsf

    # def test_real(self):
    #     # Test real pole placement using KNV and YT0 algorithm and example 1 in
    #     # section 4 of the reference publication (see place_poles docstring)
    #     A = np.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
    #                   0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
    #                   1.343, -2.104]).reshape(4, 4)
    #     B = np.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146,0]).reshape(4, 2)
    #     P = np.array([-0.2, -0.5, -5.0566, -8.6659])

    #     # Check that both KNV and YT compute correct K matrix
    #     self._check(A, B, P, method='KNV0')
    #     self._check(A, B, P, method='YT')

    #     # Try to reach the specific case in _YT_real where two singular
    #     # values are almost equal. This is to improve code coverage but I
    #     # have no way to be sure this code is really reached

    #     self._check(A, B, (2,2,3,3))

    # def test_complex(self):
    #     # Test complex pole placement on a linearized car model, taken from L.
    #     # Jaulin, Automatique pour la robotique, Cours et Exercices, iSTE
    #     # editions p 184/185
    #     A = np.array([0,7,0,0,0,0,0,7/3.,0,0,0,0,0,0,0,0]).reshape(4,4)
    #     B = np.array([0,0,0,0,1,0,0,1]).reshape(4,2)
    #     # Test complex poles on YT
    #     P = np.array([-3, -1, -2-1j, -2+1j])
    #     self._check(A, B, P)

    #     # Try to reach the specific case in _YT_complex where two singular
    #     # values are almost equal. This is to improve code coverage but I
    #     # have no way to be sure this code is really reached

    #     P = [0-1e-6j,0+1e-6j,-10,10]
    #     self._check(A, B, P, maxiter=1000)

    #     # Try to reach the specific case in _YT_complex where the rank two
    #     # update yields two null vectors. This test was found via Monte Carlo.

    #     A = np.array(
    #                 [-2148,-2902, -2267, -598, -1722, -1829, -165, -283, -2546,
    #                -167, -754, -2285, -543, -1700, -584, -2978, -925, -1300,
    #                -1583, -984, -386, -2650, -764, -897, -517, -1598, 2, -1709,
    #                -291, -338, -153, -1804, -1106, -1168, -867, -2297]
    #                ).reshape(6,6)

    #     B = np.array(
    #                 [-108, -374, -524, -1285, -1232, -161, -1204, -672, -637,
    #                  -15, -483, -23, -931, -780, -1245, -1129, -1290, -1502,
    #                  -952, -1374, -62, -964, -930, -939, -792, -756, -1437,
    #                  -491, -1543, -686]
    #                  ).reshape(6,5)
    #     P = [-25.-29.j, -25.+29.j, 31.-42.j, 31.+42.j, 33.-41.j, 33.+41.j]
    #     self._check(A, B, P)

    #     # Use a lot of poles to go through all cases for update_order
    #     # in _YT_loop

    #     big_A = np.ones((11,11))-np.eye(11)
    #     big_B = np.ones((11,10))-np.diag([1]*10,1)[:,1:]
    #     big_A[:6,:6] = A
    #     big_B[:6,:5] = B

    #     P = [-10,-20,-30,40,50,60,70,-20-5j,-20+5j,5+3j,5-3j]
    #     self._check(big_A, big_B, P)

    #     #check with only complex poles and only real poles
    #     P = [-10,-20,-30,-40,-50,-60,-70,-80,-90,-100]
    #     self._check(big_A[:-1,:-1], big_B[:-1,:-1], P)
    #     P = [-10+10j,-20+20j,-30+30j,-40+40j,-50+50j,
    #          -10-10j,-20-20j,-30-30j,-40-40j,-50-50j]
    #     self._check(big_A[:-1,:-1], big_B[:-1,:-1], P)

    #     # need a 5x5 array to ensure YT handles properly when there
    #     # is only one real pole and several complex
    #     A = np.array([0,7,0,0,0,0,0,7/3.,0,0,0,0,0,0,0,0,
    #                   0,0,0,5,0,0,0,0,9]).reshape(5,5)
    #     B = np.array([0,0,0,0,1,0,0,1,2,3]).reshape(5,2)
    #     P = np.array([-2, -3+1j, -3-1j, -1+1j, -1-1j])
    #     place_poles(A, B, P)

    #     # same test with an odd number of real poles > 1
    #     # this is another specific case of YT
    #     P = np.array([-2, -3, -4, -1+1j, -1-1j])
    #     self._check(A, B, P)

    # def test_tricky_B(self):
    #     # check we handle as we should the 1 column B matrices and
    #     # n column B matrices (with n such as shape(A)=(n, n))
    #     A = np.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
    #                   0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
    #                   1.343, -2.104]).reshape(4, 4)
    #     B = np.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0, 1, 2, 3, 4,
    #                   5, 6, 7, 8]).reshape(4, 4)

    #     # KNV or YT are not called here, it's a specific case with only
    #     # one unique solution
    #     P = np.array([-0.2, -0.5, -5.0566, -8.6659])
    #     fsf = self._check(A, B, P)
    #     # rtol and nb_iter should be set to np.nan as the identity can be
    #     # used as transfer matrix
    #     assert_equal(fsf.rtol, np.nan)
    #     assert_equal(fsf.nb_iter, np.nan)

    #     # check with complex poles too as they trigger a specific case in
    #     # the specific case :-)
    #     P = np.array((-2+1j,-2-1j,-3,-2))
    #     fsf = self._check(A, B, P)
    #     assert_equal(fsf.rtol, np.nan)
    #     assert_equal(fsf.nb_iter, np.nan)

    #     #now test with a B matrix with only one column (no optimisation)
    #     B = B[:,0].reshape(4,1)
    #     P = np.array((-2+1j,-2-1j,-3,-2))
    #     fsf = self._check(A, B, P)

    #     #  we can't optimize anything, check they are set to 0 as expected
    #     assert_equal(fsf.rtol, 0)
    #     assert_equal(fsf.nb_iter, 0)

    # def test_errors(self):
    #     # Test input mistakes from user
    #     A = np.array([0,7,0,0,0,0,0,7/3.,0,0,0,0,0,0,0,0]).reshape(4,4)
    #     B = np.array([0,0,0,0,1,0,0,1]).reshape(4,2)

    #     # #should fail as the method keyword is invalid
    #     # assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4),
    #     #               method="foo")

    #     # #should fail as poles are not 1D array
    #     # assert_raises(ValueError, place_poles, A, B,
    #     #               np.array((-2.1,-2.2,-2.3,-2.4)).reshape(4,1))

    #     # #should fail as A is not a 2D array
    #     # assert_raises(ValueError, place_poles, A[:,:,np.newaxis], B,
    #     #               (-2.1,-2.2,-2.3,-2.4))

    #     # #should fail as B is not a 2D array
    #     # assert_raises(ValueError, place_poles, A, B[:,:,np.newaxis],
    #     #               (-2.1,-2.2,-2.3,-2.4))

    #     # #should fail as there are too many poles
    #     # assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4,-3))

    #     # #should fail as there are not enough poles
    #     # assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3))

    #     # #should fail as the rtol is greater than 1
    #     # assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4),
    #     #               rtol=42)

    #     # #should fail as maxiter is smaller than 1
    #     # assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4),
    #     #               maxiter=-42)

    #     # # should fail as rank(B) is two
    #     # assert_raises(ValueError, place_poles, A, B, (-2,-2,-2,-2))

    #     # #unctrollable system
    #     # assert_raises(ValueError, place_poles, np.ones((4,4)),
    #     #               np.ones((4,2)), (1,2,3,4))

    #     # Should not raise ValueError as the poles can be placed but should
    #     # raise a warning as the convergence is not reached
    #     # with warnings.catch_warnings(record=True) as w:
    #     #     warnings.simplefilter("always")
    #     #     fsf = place_poles(A, B, (-1,-2,-3,-4), rtol=1e-16, maxiter=42)
    #     #     assert_(len(w) == 1)
    #     #     assert_(issubclass(w[-1].category, UserWarning))
    #     #     assert_("Convergence was not reached after maxiter iterations"
    #     #             in str(w[-1].message))
    #     #     assert_equal(fsf.nb_iter, 42)

    #     # # should fail as a complex misses its conjugate
    #     # assert_raises(ValueError, place_poles, A, B, (-2+1j,-2-1j,-2+3j,-2))

    #     # # should fail as A is not square
    #     # assert_raises(ValueError, place_poles, A[:,:3], B, (-2,-3,-4,-5))

    #     # # should fail as B has not the same number of lines as A
    #     # assert_raises(ValueError, place_poles, A, B[:3,:], (-2,-3,-4,-5))

    #     # # should fail as KNV0 does not support complex poles
    #     # assert_raises(ValueError, place_poles, A, B,
    #     #               (-2+1j,-2-1j,-2+3j,-2-3j), method="KNV0")

