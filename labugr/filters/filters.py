
from __future__ import division, print_function, absolute_import

#Algunas funciones utilizan numpy y otras np
import warnings
import numpy
import numpy as np
from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
                   resize, pi, absolute, logspace, r_, sqrt, tan, log10,
                   arctan, arcsinh, sin, exp, cosh, arccosh, ceil, conjugate,
                   zeros, sinh, append, concatenate, prod, ones, array,
                   mintypecode)
from numpy.polynomial.polynomial import (polyval as npp_polyval, polyvalfromroots)

__all__ = ['freqs', 'freqz', 'freqs_zpk', 'freqz_zpk', 'normalize', 'tf2zpk',
           'zpk2tf']


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

def normalize(b, a):
    """Normalize numerator/denominator of a continuous-time transfer function.

    If values of `b` are too close to 0, they are removed. In that case, a
    BadCoefficients warning is emitted.

    Parameters
    ----------
    b: array_like
        Numerator of the transfer function. Can be a 2d array to normalize
        multiple transfer functions.
    a: array_like
        Denominator of the transfer function. At most 1d.

    Returns
    -------
    num: array
        The numerator of the normalized transfer function. At least a 1d
        array. A 2d-array if the input `num` is a 2d array.
    den: 1d-array
        The denominator of the normalized transfer function.

    Notes
    -----
    Coefficients for both the numerator and denominator should be specified in
    descending exponent order (e.g., ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``).
    """
    num, den = b, a

    den = np.atleast_1d(den)
    num = np.atleast_2d(_align_nums(num))

    if den.ndim != 1:
        raise ValueError("Denominator polynomial must be rank-1 array.")
    if num.ndim > 2:
        raise ValueError("Numerator polynomial must be rank-1 or"
                         " rank-2 array.")
    if np.all(den == 0):
        raise ValueError("Denominator must have at least on nonzero element.")

    # Trim leading zeros in denominator, leave at least one.
    den = np.trim_zeros(den, 'f')

    # Normalize transfer function
    num, den = num / den[0], den / den[0]

    # Count numerator columns that are all zero
    leading_zeros = 0
    for col in num.T:
        if np.allclose(col, 0, atol=1e-14):
            leading_zeros += 1
        else:
            break

    # Trim leading zeros of numerator
    if leading_zeros > 0:
        warnings.warn("Badly conditioned filter coefficients (numerator): the "
                      "results may be meaningless", BadCoefficients)
        # Make sure at least one column remains
        if leading_zeros == num.shape[1]:
            leading_zeros -= 1
        num = num[:, leading_zeros:]

    # Squeeze first dimension if singular
    if num.shape[0] == 1:
        num = num[0, :]

    return num, den

def tf2zpk(b, a):
    r"""Return zero, pole, gain (z, p, k) representation from a numerator,
    denominator representation of a linear filter.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.

    Returns
    -------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Notes
    -----
    If some values of `b` are too close to 0, they are removed. In that case,
    a BadCoefficients warning is emitted.

    The `b` and `a` arrays are interpreted as coefficients for positive,
    descending powers of the transfer function variable.  So the inputs
    :math:`b = [b_0, b_1, ..., b_M]` and :math:`a =[a_0, a_1, ..., a_N]`
    can represent an analog filter of the form:

    .. math::

        H(s) = \frac
        {b_0 s^M + b_1 s^{(M-1)} + \cdots + b_M}
        {a_0 s^N + a_1 s^{(N-1)} + \cdots + a_N}

    or a discrete-time filter of the form:

    .. math::

        H(z) = \frac
        {b_0 z^M + b_1 z^{(M-1)} + \cdots + b_M}
        {a_0 z^N + a_1 z^{(N-1)} + \cdots + a_N}

    This "positive powers" form is found more commonly in controls
    engineering.  If `M` and `N` are equal (which is true for all filters
    generated by the bilinear transform), then this happens to be equivalent
    to the "negative powers" discrete-time form preferred in DSP:

    .. math::

        H(z) = \frac
        {b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}
        {a_0 + a_1 z^{-1} + \cdots + a_N z^{-N}}

    Although this is true for common filters, remember that this is not true
    in the general case.  If `M` and `N` are not equal, the discrete-time
    transfer function coefficients must first be converted to the "positive
    powers" form before finding the poles and zeros.

    """
    b, a = normalize(b, a)
    b = (b + 0.0) / a[0]
    a = (a + 0.0) / a[0]
    k = b[0]
    b /= b[0]
    z = roots(b)
    p = roots(a)
    return z, p, k

def zpk2tf(z, p, k):
    """
    Return polynomial transfer function representation from zeros and poles

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    """
    z = atleast_1d(z)
    k = atleast_1d(k)
    if len(z.shape) > 1:
        temp = poly(z[0])
        b = zeros((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * poly(z[i])
    else:
        b = k * poly(z)
    a = atleast_1d(poly(p))

    # Use real output if possible.  Copied from numpy.poly, since
    # we can't depend on a specific version of numpy.
    if issubclass(b.dtype.type, numpy.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = numpy.asarray(z, complex)
        pos_roots = numpy.compress(roots.imag > 0, roots)
        neg_roots = numpy.conjugate(numpy.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if numpy.all(numpy.sort_complex(neg_roots) ==
                         numpy.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, numpy.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = numpy.asarray(p, complex)
        pos_roots = numpy.compress(roots.imag > 0, roots)
        neg_roots = numpy.conjugate(numpy.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if numpy.all(numpy.sort_complex(neg_roots) ==
                         numpy.sort_complex(pos_roots)):
                a = a.real.copy()

    return b, a

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

def _align_nums(nums):
    """Aligns the shapes of multiple numerators.

    Given an array of numerator coefficient arrays [[a_1, a_2,...,
    a_n],..., [b_1, b_2,..., b_m]], this function pads shorter numerator
    arrays with zero's so that all numerators have the same length. Such
    alignment is necessary for functions like 'tf2ss', which needs the
    alignment when dealing with SIMO transfer functions.

    Parameters
    ----------
    nums: array_like
        Numerator or list of numerators. Not necessarily with same length.

    Returns
    -------
    nums: array
        The numerator. If `nums` input was a list of numerators then a 2d
        array with padded zeros for shorter numerators is returned. Otherwise
        returns ``np.asarray(nums)``.
    """
    try:
        # The statement can throw a ValueError if one
        # of the numerators is a single digit and another
        # is array-like e.g. if nums = [5, [1, 2, 3]]
        nums = asarray(nums)

        if not np.issubdtype(nums.dtype, np.number):
            raise ValueError("dtype of numerator is non-numeric")

        return nums

    except ValueError:
        nums = [np.atleast_1d(num) for num in nums]
        max_width = max(num.size for num in nums)

        # pre-allocate
        aligned_nums = np.zeros((len(nums), max_width))

        # Create numerators with padded zeros
        for index, num in enumerate(nums):
            aligned_nums[index, -num.size:] = num

        return aligned_nums

class BadCoefficients(UserWarning):
    """Warning about badly conditioned filter coefficients"""
    pass
