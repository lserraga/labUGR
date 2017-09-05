from __future__ import division, print_function, absolute_import

import numpy as np
from math import ceil, log
from numpy.fft import irfft
from numpy.linalg import pinv


__all__ = ['firwin', 'firwin2', 'firls']

def firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True,
           scale=True, nyq=1.0):
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response
    filter.  The filter will have linear phase; it will be Type I if
    `numtaps` is odd and Type II if `numtaps` is even.

    Type II filters always have zero response at the Nyquist rate, so a
    ValueError exception is raised if firwin is called with `numtaps` even and
    having a passband whose right end is at the Nyquist rate.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients, i.e. the filter
        order + 1).  `numtaps` must be even if a passband includes the
        Nyquist frequency.
    cutoff : float or 1D array_like
        Cutoff frequency of filter (expressed in the same units as `nyq`)
        OR an array of cutoff frequencies (that is, band edges). In the
        latter case, the frequencies in `cutoff` should be positive and
        monotonically increasing between 0 and `nyq`.  The values 0 and
        `nyq` must not be included in `cutoff`.
    width : float or None, optional
        If `width` is not None, then assume it is the approximate width
        of the transition region (expressed in the same units as `nyq`)
        for use in Kaiser FIR filter design.  In this case, the `window`
        argument is ignored.
    window : string or tuple of string and parameter values, optional
        Desired window to use. See `scipy.signal.get_window` for a list
        of windows and required parameters.
    pass_zero : bool, optional
        If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
        Otherwise the DC gain is 0.
    scale : bool, optional
        Set to True to scale the coefficients so that the frequency
        response is exactly unity at a certain frequency.
        That frequency is either:

        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
          is True)
        - `nyq` (the Nyquist rate) if the first passband ends at
          `nyq` (i.e the filter is a single band highpass filter);
          center of first passband otherwise

    nyq : float, optional
        Nyquist frequency.  Each frequency in `cutoff` must be between 0
        and `nyq`.

    Returns
    -------
    h : (numtaps,) ndarray
        Coefficients of length `numtaps` FIR filter.

    Raises
    ------
    ValueError
        If any value in `cutoff` is less than or equal to 0 or greater
        than or equal to `nyq`, if the values in `cutoff` are not strictly
        monotonically increasing, or if `numtaps` is even but a passband
        includes the Nyquist frequency.

    See also
    --------
    firwin2
    firls
    minimum_phase
    remez

    Examples
    --------
    Low-pass from 0 to f:

    >>> from scipy import signal
    >>> numtaps = 3
    >>> f = 0.1
    >>> signal.firwin(numtaps, f)
    array([ 0.06799017,  0.86401967,  0.06799017])

    Use a specific window function:

    >>> signal.firwin(numtaps, f, window='nuttall')
    array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])

    High-pass ('stop' from 0 to f):

    >>> signal.firwin(numtaps, f, pass_zero=False)
    array([-0.00859313,  0.98281375, -0.00859313])

    Band-pass:

    >>> f1, f2 = 0.1, 0.2
    >>> signal.firwin(numtaps, [f1, f2], pass_zero=False)
    array([ 0.06301614,  0.88770441,  0.06301614])

    Band-stop:

    >>> signal.firwin(numtaps, [f1, f2])
    array([-0.00801395,  1.0160279 , -0.00801395])

    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):

    >>> f3, f4 = 0.3, 0.4
    >>> signal.firwin(numtaps, [f1, f2, f3, f4])
    array([-0.01376344,  1.02752689, -0.01376344])

    Multi-band (passbands are [f1, f2] and [f3,f4]):

    >>> signal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)
    array([ 0.04890915,  0.91284326,  0.04890915])

    """

    # The major enhancements to this function added in November 2010 were
    # developed by Tom Krauss (see ticket #902).

    cutoff = np.atleast_1d(cutoff) / float(nyq)

    # Check for invalid input.
    if cutoff.ndim > 1:
        raise ValueError("The cutoff argument must be at most "
                         "one-dimensional.")
    if cutoff.size == 0:
        raise ValueError("At least one cutoff frequency must be given.")
    if cutoff.min() <= 0 or cutoff.max() >= 1:
        raise ValueError("Invalid cutoff frequency: frequencies must be "
                         "greater than 0 and less than nyq.")
    if np.any(np.diff(cutoff) <= 0):
        raise ValueError("Invalid cutoff frequencies: the frequencies "
                         "must be strictly increasing.")

    if width is not None:
        # A width was given.  Find the beta parameter of the Kaiser window
        # and set `window`.  This overrides the value of `window` passed in.
        atten = kaiser_atten(numtaps, float(width) / nyq)
        beta = kaiser_beta(atten)
        window = ('kaiser', beta)

    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError("A filter with an even number of coefficients must "
                         "have zero response at the Nyquist rate.")

    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    cutoff = np.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))

    # `bands` is a 2D array; each row gives the left and right edges of
    # a passband.
    bands = cutoff.reshape(-1, 2)

    # Build up the coefficients.
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps) - alpha
    h = 0
    for left, right in bands:
        h += right * np.sinc(right * m)
        h -= left * np.sinc(left * m)

    # Get and apply the window function.
    from labugr.signal.windows import get_window
    win = get_window(window, numtaps, fftbins=False)
    h *= win

    # Now handle scaling if desired.
    if scale:
        # Get the first passband.
        left, right = bands[0]
        if left == 0:
            scale_frequency = 0.0
        elif right == 1:
            scale_frequency = 1.0
        else:
            scale_frequency = 0.5 * (left + right)
        c = np.cos(np.pi * m * scale_frequency)
        s = np.sum(h * c)
        h /= s

    return h


def firwin2(numtaps, freq, gain, nfreqs=None, window='hamming', nyq=1.0,
            antisymmetric=False):
    """
    FIR filter design using the window method.

    From the given frequencies `freq` and corresponding gains `gain`,
    this function constructs an FIR filter with linear phase and
    (approximately) the given frequency response.

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be less than
        `nfreqs`.
    freq : array_like, 1D
        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
        Nyquist.  The Nyquist frequency can be redefined with the argument
        `nyq`.
        The values in `freq` must be nondecreasing.  A value can be repeated
        once to implement a discontinuity.  The first value in `freq` must
        be 0, and the last value must be `nyq`.
    gain : array_like
        The filter gains at the frequency sampling points. Certain
        constraints to gain values, depending on the filter type, are applied,
        see Notes for details.
    nfreqs : int, optional
        The size of the interpolation mesh used to construct the filter.
        For most efficient behavior, this should be a power of 2 plus 1
        (e.g, 129, 257, etc).  The default is one more than the smallest
        power of 2 that is not less than `numtaps`.  `nfreqs` must be greater
        than `numtaps`.
    window : string or (string, float) or float, or None, optional
        Window function to use. Default is "hamming".  See
        `scipy.signal.get_window` for the complete list of possible values.
        If None, no window function is applied.
    nyq : float, optional
        Nyquist frequency.  Each frequency in `freq` must be between 0 and
        `nyq` (inclusive).
    antisymmetric : bool, optional
        Whether resulting impulse response is symmetric/antisymmetric.
        See Notes for more details.

    Returns
    -------
    taps : ndarray
        The filter coefficients of the FIR filter, as a 1-D array of length
        `numtaps`.

    See also
    --------
    firls
    firwin
    minimum_phase
    remez

    Notes
    -----
    From the given set of frequencies and gains, the desired response is
    constructed in the frequency domain.  The inverse FFT is applied to the
    desired response to create the associated convolution kernel, and the
    first `numtaps` coefficients of this kernel, scaled by `window`, are
    returned.

    The FIR filter will have linear phase. The type of filter is determined by
    the value of 'numtaps` and `antisymmetric` flag.
    There are four possible combinations:

       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced
       - even `numtaps`, `antisymmetric` is False, type II filter is produced
       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced
       - even `numtaps`, `antisymmetric` is True, type IV filter is produced

    Magnitude response of all but type I filters are subjects to following
    constraints:

       - type II  -- zero at the Nyquist frequency
       - type III -- zero at zero and Nyquist frequencies
       - type IV  -- zero at zero frequency

    .. versionadded:: 0.9.0

    References
    ----------
    .. [1] Oppenheim, A. V. and Schafer, R. W., "Discrete-Time Signal
       Processing", Prentice-Hall, Englewood Cliffs, New Jersey (1989).
       (See, for example, Section 7.4.)

    .. [2] Smith, Steven W., "The Scientist and Engineer's Guide to Digital
       Signal Processing", Ch. 17. http://www.dspguide.com/ch17/1.htm

    Examples
    --------
    A lowpass FIR filter with a response that is 1 on [0.0, 0.5], and
    that decreases linearly on [0.5, 1.0] from 1 to 0:

    >>> from scipy import signal
    >>> taps = signal.firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    >>> print(taps[72:78])
    [-0.02286961 -0.06362756  0.57310236  0.57310236 -0.06362756 -0.02286961]

    """

    if len(freq) != len(gain):
        raise ValueError('freq and gain must be of same length.')

    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError(('ntaps must be less than nfreqs, but firwin2 was '
                          'called with ntaps=%d and nfreqs=%s') %
                         (numtaps, nfreqs))

    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError('freq must start with 0 and end with `nyq`.')
    d = np.diff(freq)
    if (d < 0).any():
        raise ValueError('The values in freq must be nondecreasing.')
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError('A value in freq must not occur more than twice.')

    if antisymmetric:
        if numtaps % 2 == 0:
            ftype = 4
        else:
            ftype = 3
    else:
        if numtaps % 2 == 0:
            ftype = 2
        else:
            ftype = 1

    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError("A Type II filter must have zero gain at the "
                         "Nyquist rate.")
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError("A Type III filter must have zero gain at zero "
                         "and Nyquist rates.")
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError("A Type IV filter must have zero gain at zero rate.")

    if nfreqs is None:
        nfreqs = 1 + 2 ** int(ceil(log(numtaps, 2)))

    # Tweak any repeated values in freq so that interp works.
    eps = np.finfo(float).eps
    for k in range(len(freq)):
        if k < len(freq) - 1 and freq[k] == freq[k + 1]:
            freq[k] = freq[k] - eps
            freq[k + 1] = freq[k + 1] + eps

    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = np.linspace(0.0, nyq, nfreqs)
    fx = np.interp(x, freq, gain)

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = np.exp(-(numtaps - 1) / 2. * 1.j * np.pi * x / nyq)
    if ftype > 2:
        shift *= 1j

    fx2 = fx * shift

    # Use irfft to compute the inverse FFT.
    out_full = irfft(fx2)

    if window is not None:
        # Create the window to apply to the filter coefficients.
        from labugr.signal.windows import get_window
        wind = get_window(window, numtaps, fftbins=False)
    else:
        wind = 1

    # Keep only the first `numtaps` coefficients in `out`, and multiply by
    # the window.
    out = out_full[:numtaps] * wind

    if ftype == 3:
        out[out.size // 2] = 0.0

    return out



def firls(numtaps, bands, desired, weight=None, nyq=1.):
    """
    FIR filter design using least-squares error minimization.

    Calculate the filter coefficients for the linear-phase finite
    impulse response (FIR) filter which has the best approximation
    to the desired frequency response described by `bands` and
    `desired` in the least squares sense (i.e., the integral of the
    weighted mean-squared error within the specified bands is
    minimized).

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be odd.
    bands : array_like
        A monotonic nondecreasing sequence containing the band edges in
        Hz. All elements must be non-negative and less than or equal to
        the Nyquist frequency given by `nyq`.
    desired : array_like
        A sequence the same size as `bands` containing the desired gain
        at the start and end point of each band.
    weight : array_like, optional
        A relative weighting to give to each band region when solving
        the least squares problem. `weight` has to be half the size of
        `bands`.
    nyq : float, optional
        Nyquist frequency. Each frequency in `bands` must be between 0
        and `nyq` (inclusive).

    Returns
    -------
    coeffs : ndarray
        Coefficients of the optimal (in a least squares sense) FIR filter.

    See also
    --------
    firwin
    firwin2
    minimum_phase
    remez

    Notes
    -----
    This implementation follows the algorithm given in [1]_.
    As noted there, least squares design has multiple advantages:

        1. Optimal in a least-squares sense.
        2. Simple, non-iterative method.
        3. The general solution can obtained by solving a linear
           system of equations.
        4. Allows the use of a frequency dependent weighting function.

    This function constructs a Type I linear phase FIR filter, which
    contains an odd number of `coeffs` satisfying for :math:`n < numtaps`:

    .. math:: coeffs(n) = coeffs(numtaps - 1 - n)

    The odd number of coefficients and filter symmetry avoid boundary
    conditions that could otherwise occur at the Nyquist and 0 frequencies
    (e.g., for Type II, III, or IV variants).

    .. versionadded:: 0.18

    References
    ----------
    .. [1] Ivan Selesnick, Linear-Phase Fir Filter Design By Least Squares.
           OpenStax CNX. Aug 9, 2005.
           http://cnx.org/contents/eb1ecb35-03a9-4610-ba87-41cd771c95f2@7

    Examples
    --------
    We want to construct a band-pass filter. Note that the behavior in the
    frequency ranges between our stop bands and pass bands is unspecified,
    and thus may overshoot depending on the parameters of our filter:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> fig, axs = plt.subplots(2)
    >>> nyq = 5.  # Hz
    >>> desired = (0, 0, 1, 1, 0, 0)
    >>> for bi, bands in enumerate(((0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 4.5, 5))):
    ...     fir_firls = signal.firls(73, bands, desired, nyq=nyq)
    ...     fir_remez = signal.remez(73, bands, desired[::2], Hz=2 * nyq)
    ...     fir_firwin2 = signal.firwin2(73, bands, desired, nyq=nyq)
    ...     hs = list()
    ...     ax = axs[bi]
    ...     for fir in (fir_firls, fir_remez, fir_firwin2):
    ...         freq, response = signal.freqz(fir)
    ...         hs.append(ax.semilogy(nyq*freq/(np.pi), np.abs(response))[0])
    ...     for band, gains in zip(zip(bands[::2], bands[1::2]), zip(desired[::2], desired[1::2])):
    ...         ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)
    ...     if bi == 0:
    ...         ax.legend(hs, ('firls', 'remez', 'firwin2'), loc='lower center', frameon=False)
    ...     else:
    ...         ax.set_xlabel('Frequency (Hz)')
    ...     ax.grid(True)
    ...     ax.set(title='Band-pass %d-%d Hz' % bands[2:4], ylabel='Magnitude')
    ...
    >>> fig.tight_layout()
    >>> plt.show()

    """  # noqa
    numtaps = int(numtaps)
    if numtaps % 2 == 0 or numtaps < 1:
        raise ValueError("numtaps must be odd and >= 1")
    M = (numtaps-1) // 2

    # normalize bands 0->1 and make it 2 columns
    nyq = float(nyq)
    if nyq <= 0:
        raise ValueError('nyq must be positive, got %s <= 0.' % nyq)
    bands = np.asarray(bands).flatten() / nyq
    if len(bands) % 2 != 0:
        raise ValueError("bands must contain frequency pairs.")
    bands.shape = (-1, 2)

    # check remaining params
    desired = np.asarray(desired).flatten()
    if bands.size != desired.size:
        raise ValueError("desired must have one entry per frequency, got %s "
                         "gains for %s frequencies."
                         % (desired.size, bands.size))
    desired.shape = (-1, 2)
    if (np.diff(bands) <= 0).any() or (np.diff(bands[:, 0]) < 0).any():
        raise ValueError("bands must be monotonically nondecreasing and have "
                         "width > 0.")
    if (bands[:-1, 1] > bands[1:, 0]).any():
        raise ValueError("bands must not overlap.")
    if (desired < 0).any():
        raise ValueError("desired must be non-negative.")
    if weight is None:
        weight = np.ones(len(desired))
    weight = np.asarray(weight).flatten()
    if len(weight) != len(desired):
        raise ValueError("weight must be the same size as the number of "
                         "band pairs (%s)." % (len(bands),))
    if (weight < 0).any():
        raise ValueError("weight must be non-negative.")

    # Set up the linear matrix equation to be solved, Qa = b

    # We can express Q(k,n) = 0.5 Q1(k,n) + 0.5 Q2(k,n)
    # where Q1(k,n)=q(k−n) and Q2(k,n)=q(k+n), i.e. a Toeplitz plus Hankel.

    # We omit the factor of 0.5 above, instead adding it during coefficient
    # calculation.

    # We also omit the 1/π from both Q and b equations, as they cancel
    # during solving.

    # We have that:
    #     q(n) = 1/π ∫W(ω)cos(nω)dω (over 0->π)
    # Using our nomalization ω=πf and with a constant weight W over each
    # interval f1->f2 we get:
    #     q(n) = W∫cos(πnf)df (0->1) = Wf sin(πnf)/πnf
    # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
    n = np.arange(numtaps)[:, np.newaxis, np.newaxis]
    q = np.dot(np.diff(np.sinc(bands * n) * bands, axis=2)[:, :, 0], weight)

    # Now we assemble our sum of Toeplitz and Hankel
    Q1 = toeplitz(q[:M+1])
    Q2 = hankel(q[:M+1], q[M:])
    Q = Q1 + Q2

    # Now for b(n) we have that:
    #     b(n) = 1/π ∫ W(ω)D(ω)cos(nω)dω (over 0->π)
    # Using our nomalization ω=πf and with a constant weight W over each
    # interval and a linear term for D(ω) we get (over each f1->f2 interval):
    #     b(n) = W ∫ (mf+c)cos(πnf)df
    #          = f(mf+c)sin(πnf)/πnf + mf**2 cos(nπf)/(πnf)**2
    # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
    n = n[:M + 1]  # only need this many coefficients here
    # Choose m and c such that we are at the start and end weights
    m = (np.diff(desired, axis=1) / np.diff(bands, axis=1))
    c = desired[:, [0]] - bands[:, [0]] * m
    b = bands * (m*bands + c) * np.sinc(bands * n)
    # Use L'Hospital's rule here for cos(nπf)/(πnf)**2 @ n=0
    b[0] -= m * bands * bands / 2.
    b[1:] += m * np.cos(n[1:] * np.pi * bands) / (np.pi * n[1:]) ** 2
    b = np.dot(np.diff(b, axis=2)[:, :, 0], weight)

    # Now we can solve the equation (use pinv because Q can be rank deficient)
    a = np.dot(pinv(Q), b)

    # make coefficients symmetric (linear phase)
    coeffs = np.hstack((a[:0:-1], 2 * a[0], a[1:]))
    return coeffs

def kaiser_beta(a):
    """Compute the Kaiser parameter `beta`, given the attenuation `a`.

    Parameters
    ----------
    a : float
        The desired attenuation in the stopband and maximum ripple in
        the passband, in dB.  This should be a *positive* number.

    Returns
    -------
    beta : float
        The `beta` parameter to be used in the formula for a Kaiser window.

    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.
    """
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    return beta


def kaiser_atten(numtaps, width):
    """Compute the attenuation of a Kaiser FIR filter.

    Given the number of taps `N` and the transition width `width`, compute the
    attenuation `a` in dB, given by Kaiser's formula:

        a = 2.285 * (N - 1) * pi * width + 7.95

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.
    width : float
        The desired width of the transition region between passband and
        stopband (or, in general, at any discontinuity) for the filter.

    Returns
    -------
    a : float
        The attenuation of the ripple, in dB.

    See Also
    --------
    kaiserord, kaiser_beta
    """
    a = 2.285 * (numtaps - 1) * np.pi * width + 7.95
    return a


def kaiserord(ripple, width):
    """
    Design a Kaiser window to limit ripple and width of transition region.

    Parameters
    ----------
    ripple : float
        Positive number specifying maximum ripple in passband (dB) and minimum
        ripple in stopband.
    width : float
        Width of transition region (normalized so that 1 corresponds to pi
        radians / sample).

    Returns
    -------
    numtaps : int
        The length of the kaiser window.
    beta : float
        The beta parameter for the kaiser window.

    See Also
    --------
    kaiser_beta, kaiser_atten

    Notes
    -----
    There are several ways to obtain the Kaiser window:

    - ``signal.kaiser(numtaps, beta, sym=True)``
    - ``signal.get_window(beta, numtaps)``
    - ``signal.get_window(('kaiser', beta), numtaps)``

    The empirical equations discovered by Kaiser are used.

    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.

    """
    A = abs(ripple)  # in case somebody is confused as to what's meant
    if A < 8:
        # Formula for N is not valid in this range.
        raise ValueError("Requested maximum ripple attentuation %f is too "
                         "small for the Kaiser formula." % A)
    beta = kaiser_beta(A)

    # Kaiser's formula (as given in Oppenheim and Schafer) is for the filter
    # order, so we have to add 1 to get the number of taps.
    numtaps = (A - 7.95) / 2.285 / (np.pi * width) + 1

    return int(ceil(numtaps)), beta

def toeplitz(c, r=None):
    """
    Construct a Toeplitz matrix.

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row.  If r is not given, ``r == conjugate(c)`` is
    assumed.

    Parameters
    ----------
    c : array_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like, optional
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.

    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

    See Also
    --------
    circulant : circulant matrix
    hankel : Hankel matrix
    solve_toeplitz : Solve a Toeplitz system.

    Notes
    -----
    The behavior when `c` or `r` is a scalar, or when `c` is complex and
    `r` is None, was changed in version 0.8.0.  The behavior in previous
    versions was undocumented and is no longer supported.

    Examples
    --------
    >>> from scipy.linalg import toeplitz
    >>> toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    >>> toeplitz([1.0, 2+3j, 4-1j])
    array([[ 1.+0.j,  2.-3.j,  4.+1.j],
           [ 2.+3.j,  1.+0.j,  2.-3.j],
           [ 4.-1.j,  2.+3.j,  1.+0.j]])

    """
    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    # Form a 1D array of values to be used in the matrix, containing a reversed
    # copy of r[1:], followed by c.
    vals = np.concatenate((r[-1:0:-1], c))
    a, b = np.ogrid[0:len(c), len(r) - 1:-1:-1]
    indx = a + b
    # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
    # that `vals[indx]` is the Toeplitz matrix.
    return vals[indx]

def hankel(c, r=None):
    """
    Construct a Hankel matrix.

    The Hankel matrix has constant anti-diagonals, with `c` as its
    first column and `r` as its last row.  If `r` is not given, then
    `r = zeros_like(c)` is assumed.

    Parameters
    ----------
    c : array_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like, optional
        Last row of the matrix. If None, ``r = zeros_like(c)`` is assumed.
        r[0] is ignored; the last row of the returned matrix is
        ``[c[-1], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.

    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Hankel matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

    See Also
    --------
    toeplitz : Toeplitz matrix
    circulant : circulant matrix

    Examples
    --------
    >>> from scipy.linalg import hankel
    >>> hankel([1, 17, 99])
    array([[ 1, 17, 99],
           [17, 99,  0],
           [99,  0,  0]])
    >>> hankel([1,2,3,4], [4,7,7,8,9])
    array([[1, 2, 3, 4, 7],
           [2, 3, 4, 7, 7],
           [3, 4, 7, 7, 8],
           [4, 7, 7, 8, 9]])

    """
    c = np.asarray(c).ravel()
    if r is None:
        r = np.zeros_like(c)
    else:
        r = np.asarray(r).ravel()
    # Form a 1D array of values to be used in the matrix, containing `c`
    # followed by r[1:].
    vals = np.concatenate((c, r[1:]))
    a, b = np.ogrid[0:len(c), 0:len(r)]
    indx = a + b
    # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
    # that `vals[indx]` is the Hankel matrix.
    return vals[indx]