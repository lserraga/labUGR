
from __future__ import division, print_function, absolute_import

#Algunas funciones utilizan numpy y otras np

import numpy
import numpy as np
from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
                   resize, pi, absolute, logspace, r_, sqrt, tan, log10,
                   arctan, arcsinh, sin, exp, cosh, arccosh, ceil, conjugate,
                   zeros, sinh, append, concatenate, prod, ones, array,
                   mintypecode)
from numpy.polynomial.polynomial import (polyval as npp_polyval, polyvalfromroots)

__all__ = ['freqs', 'freqz', 'freqs_zpk', 'freqz_zpk']


def freqs(b, a, worN=None, plot=None):
    """
    Compute frequency response of analog filter.

    Given the M-order numerator `b` and N-order denominator `a` of an analog
    filter, compute its frequency response::

             b[0]*(jw)**M + b[1]*(jw)**(M-1) + ... + b[M]
     H(w) = ----------------------------------------------
             a[0]*(jw)**N + a[1]*(jw)**(N-1) + ... + a[N]

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations).  If a single
        integer, then compute at that many frequencies.  Otherwise, compute the
        response at the angular frequencies (e.g. rad/s) given in `worN`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqs`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    freqz : Compute the frequency response of a digital filter.

    Notes
    -----
    Using Matplotlib's "plot" function as the callable for `plot` produces
    unexpected results,  this plots the real part of the complex transfer
    function, not the magnitude.  Try ``lambda w, h: plot(w, abs(h))``.

    Examples
    --------
    >>> from scipy.signal import freqs, iirfilter

    >>> b, a = iirfilter(4, [1, 10], 1, 60, analog=True, ftype='cheby1')

    >>> w, h = freqs(b, a, worN=np.logspace(-1, 2, 1000))

    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Amplitude response [dB]')
    >>> plt.grid()
    >>> plt.show()

    """
    if worN is None:
        w = findfreqs(b, a, 200)
    elif isinstance(worN, int):
        N = worN
        w = findfreqs(b, a, N)
    else:
        w = worN
    w = atleast_1d(w)
    s = 1j * w
    h = polyval(b, s) / polyval(a, s)
    if plot is not None:
        plot(w, h)

    return w, h

def freqz(b, a=1, worN=None, whole=False, plot=None):
    """
    Compute the frequency response of a digital filter.

    Given the M-order numerator `b` and N-order denominator `a` of a digital
    filter, compute its frequency response::

                 jw               -jw               -jwM
        jw    B(e  )  b[0] + b[1]e    + .... + b[M]e
     H(e  ) = ---- = -----------------------------------
                 jw               -jw               -jwN
              A(e  )  a[0] + a[1]e    + .... + a[N]e

    Parameters
    ----------
    b : array_like
        numerator of a linear filter
    a : array_like
        denominator of a linear filter
    worN : {None, int, array_like}, optional
        If None (default), then compute at 512 frequencies equally spaced
        around the unit circle.
        If a single integer, then compute at that many frequencies.
        If an array_like, compute the response at the frequencies given (in
        radians/sample).
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        pi radians/sample (upper-half of unit-circle).  If `whole` is True,
        compute frequencies from 0 to 2*pi radians/sample.
    plot : callable
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqz`.

    Returns
    -------
    w : ndarray
        The normalized frequencies at which `h` was computed, in
        radians/sample.
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    sosfreqz

    Notes
    -----
    Using Matplotlib's "plot" function as the callable for `plot` produces
    unexpected results,  this plots the real part of the complex transfer
    function, not the magnitude.  Try ``lambda w, h: plot(w, abs(h))``.

    Examples
    --------
    >>> from scipy import signal
    >>> b = signal.firwin(80, 0.5, window=('kaiser', 8))
    >>> w, h = signal.freqz(b)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.title('Digital filter frequency response')
    >>> ax1 = fig.add_subplot(111)

    >>> plt.plot(w, 20 * np.log10(abs(h)), 'b')
    >>> plt.ylabel('Amplitude [dB]', color='b')
    >>> plt.xlabel('Frequency [rad/sample]')

    >>> ax2 = ax1.twinx()
    >>> angles = np.unwrap(np.angle(h))
    >>> plt.plot(w, angles, 'g')
    >>> plt.ylabel('Angle (radians)', color='g')
    >>> plt.grid()
    >>> plt.axis('tight')
    >>> plt.show()

    """
    b, a = map(atleast_1d, (b, a))
    if whole:
        lastpoint = 2 * pi
    else:
        lastpoint = pi
    if worN is None:
        N = 512
        w = numpy.linspace(0, lastpoint, N, endpoint=False)
    elif isinstance(worN, int):
        N = worN
        w = numpy.linspace(0, lastpoint, N, endpoint=False)
    else:
        w = worN
    w = atleast_1d(w)
    zm1 = exp(-1j * w)
    h = polyval(b[::-1], zm1) / polyval(a[::-1], zm1)
    if plot is not None:
        plot(w, h)

    return w, h

def freqs_zpk(z, p, k, worN=None):
    """
    Compute frequency response of analog filter.

    Given the zeros `z`, poles `p`, and gain `k` of a filter, compute its
    frequency response::

                (jw-z[0]) * (jw-z[1]) * ... * (jw-z[-1])
     H(w) = k * ----------------------------------------
                (jw-p[0]) * (jw-p[1]) * ... * (jw-p[-1])

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations).  If a single
        integer, then compute at that many frequencies.  Otherwise, compute the
        response at the angular frequencies (e.g. rad/s) given in `worN`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqz : Compute the frequency response of a digital filter in TF form
    freqz_zpk : Compute the frequency response of a digital filter in ZPK form

    Notes
    -----
    .. versionadded: 0.19.0

    Examples
    --------
    >>> from scipy.signal import freqs_zpk, iirfilter

    >>> z, p, k = iirfilter(4, [1, 10], 1, 60, analog=True, ftype='cheby1',
    ...                     output='zpk')

    >>> w, h = freqs_zpk(z, p, k, worN=np.logspace(-1, 2, 1000))

    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Amplitude response [dB]')
    >>> plt.grid()
    >>> plt.show()

    """
    k = np.asarray(k)
    if k.size > 1:
        raise ValueError('k must be a single scalar gain')

    if worN is None:
        w = findfreqs(z, p, 200, kind='zp')
    elif isinstance(worN, int):
        N = worN
        w = findfreqs(z, p, N, kind='zp')
    else:
        w = worN

    w = atleast_1d(w)
    s = 1j * w
    num = polyvalfromroots(s, z)
    den = polyvalfromroots(s, p)
    h = k * num/den
    return w, h

def freqz_zpk(z, p, k, worN=None, whole=False):
    r"""
    Compute the frequency response of a digital filter in ZPK form.

    Given the Zeros, Poles and Gain of a digital filter, compute its frequency
    response::

    :math:`H(z)=k \prod_i (z - Z[i]) / \prod_j (z - P[j])`

    where :math:`k` is the `gain`, :math:`Z` are the `zeros` and :math:`P` are
    the `poles`.

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If None (default), then compute at 512 frequencies equally spaced
        around the unit circle.
        If a single integer, then compute at that many frequencies.
        If an array_like, compute the response at the frequencies given (in
        radians/sample).
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        pi radians/sample (upper-half of unit-circle).  If `whole` is True,
        compute frequencies from 0 to 2*pi radians/sample.

    Returns
    -------
    w : ndarray
        The normalized frequencies at which `h` was computed, in
        radians/sample.
    h : ndarray
        The frequency response.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqs_zpk : Compute the frequency response of an analog filter in ZPK form
    freqz : Compute the frequency response of a digital filter in TF form

    Notes
    -----
    .. versionadded: 0.19.0

    Examples
    --------
    >>> from scipy import signal
    >>> z, p, k = signal.butter(4, 0.2, output='zpk')
    >>> w, h = signal.freqz_zpk(z, p, k)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.title('Digital filter frequency response')
    >>> ax1 = fig.add_subplot(111)

    >>> plt.plot(w, 20 * np.log10(abs(h)), 'b')
    >>> plt.ylabel('Amplitude [dB]', color='b')
    >>> plt.xlabel('Frequency [rad/sample]')

    >>> ax2 = ax1.twinx()
    >>> angles = np.unwrap(np.angle(h))
    >>> plt.plot(w, angles, 'g')
    >>> plt.ylabel('Angle (radians)', color='g')
    >>> plt.grid()
    >>> plt.axis('tight')
    >>> plt.show()

    """
    z, p = map(atleast_1d, (z, p))
    if whole:
        lastpoint = 2 * pi
    else:
        lastpoint = pi
    if worN is None:
        N = 512
        w = numpy.linspace(0, lastpoint, N, endpoint=False)
    elif isinstance(worN, int):
        N = worN
        w = numpy.linspace(0, lastpoint, N, endpoint=False)
    else:
        w = worN
    w = atleast_1d(w)
    zm1 = exp(1j * w)
    h = k * polyvalfromroots(zm1, z) / polyvalfromroots(zm1, p)

    return w, h

def findfreqs(num, den, N, kind='ba'):
    """
    Find array of frequencies for computing the response of an analog filter.

    Parameters
    ----------
    num, den : array_like, 1-D
        The polynomial coefficients of the numerator and denominator of the
        transfer function of the filter or LTI system, where the coefficients
        are ordered from highest to lowest degree. Or, the roots  of the
        transfer function numerator and denominator (i.e. zeroes and poles).
    N : int
        The length of the array to be computed.
    kind : str {'ba', 'zp'}, optional
        Specifies whether the numerator and denominator are specified by their
        polynomial coefficients ('ba'), or their roots ('zp').

    Returns
    -------
    w : (N,) ndarray
        A 1-D array of frequencies, logarithmically spaced.

    Examples
    --------
    Find a set of nine frequencies that span the "interesting part" of the
    frequency response for the filter with the transfer function

        H(s) = s / (s^2 + 8s + 25)

    >>> from scipy import signal
    >>> signal.findfreqs([1, 0], [1, 8, 25], N=9)
    array([  1.00000000e-02,   3.16227766e-02,   1.00000000e-01,
             3.16227766e-01,   1.00000000e+00,   3.16227766e+00,
             1.00000000e+01,   3.16227766e+01,   1.00000000e+02])
    """
    if kind == 'ba':
        ep = atleast_1d(roots(den)) + 0j
        tz = atleast_1d(roots(num)) + 0j
    elif kind == 'zp':
        ep = atleast_1d(den) + 0j
        tz = atleast_1d(num) + 0j
    else:
        raise ValueError("input must be one of {'ba', 'zp'}")

    if len(ep) == 0:
        ep = atleast_1d(-1000) + 0j

    ez = r_['-1',
            numpy.compress(ep.imag >= 0, ep, axis=-1),
            numpy.compress((abs(tz) < 1e5) & (tz.imag >= 0), tz, axis=-1)]

    integ = abs(ez) < 1e-10
    hfreq = numpy.around(numpy.log10(numpy.max(3 * abs(ez.real + integ) +
                                               1.5 * ez.imag)) + 0.5)
    lfreq = numpy.around(numpy.log10(0.1 * numpy.min(abs(real(ez + integ)) +
                                                     2 * ez.imag)) - 0.5)

    w = logspace(lfreq, hfreq, N)
    return w
