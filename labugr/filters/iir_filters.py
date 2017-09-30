from __future__ import division, print_function, absolute_import

import numpy
from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
                   resize, pi, absolute, logspace, r_, sqrt, tan, log10,
                   arctan, arcsinh, sin, exp, cosh, arccosh, ceil, conjugate,
                   zeros, sinh, append, concatenate, prod, ones, array,
                   mintypecode)
from .filters import *
from .helpers import fminbound, fmin, newton
from mpmath import ellipk, ellipfun, besselk
import numpy as np


__all__ = ['iirfilter', 'cheb1ord', 'cheb2ord', 'buttord']

def iirfilter(N, Wn, rp=None, rs=None, btype='band', analog=False,
              ftype='butter', output='ba'):

    """
    IIR digital and analog filter design given order and critical points.

    Design an Nth-order digital or analog filter and return the filter
    coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple
        in the passband. (dB)
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation
        in the stop band. (dB)
    btype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        The type of filter.  Default is 'bandpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'

    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba') or pole-zero ('zpk').
        Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirdesign : General filter design using passband and stopband spec

    Notes
    -----
    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Generate a 17th-order Chebyshev II bandpass filter and plot the frequency
    response:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.iirfilter(17, [50, 200], rs=60, btype='band',
    ...                         analog=True, ftype='cheby2')
    >>> w, h = signal.freqs(b, a, 1000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.semilogx(w, 20 * np.log10(abs(h)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [radians / second]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()

    """
    
    ftype, btype, output = [x.lower() for x in (ftype, btype, output)]
    Wn = asarray(Wn)
    try:
        btype = band_dict[btype]
    except KeyError:
        raise ValueError("'%s' is an invalid bandtype for filter." % btype)

    try:
        typefunc = filter_dict[ftype][0]
    except KeyError:
        raise ValueError("'%s' is not a valid basic IIR filter." % ftype)

    if output not in ['ba', 'zpk']:
        raise ValueError("'%s' is not a valid output form." % output)

    if rp is not None and rp < 0:
        raise ValueError("passband ripple (rp) must be positive")

    if rs is not None and rs < 0:
        raise ValueError("stopband attenuation (rs) must be positive")

    # Get analog lowpass prototype
    if typefunc == buttap:
        z, p, k = typefunc(N)
    # elif typefunc == besselap:
    #     z, p, k = typefunc(N, norm=bessel_norms[ftype])
    elif typefunc == cheb1ap:
        if rp is None:
            raise ValueError("passband ripple (rp) must be provided to "
                             "design a Chebyshev I filter.")
        z, p, k = typefunc(N, rp)
    elif typefunc == cheb2ap:
        if rs is None:
            raise ValueError("stopband attenuation (rs) must be provided to "
                             "design an Chebyshev II filter.")
        z, p, k = typefunc(N, rs)
    # elif typefunc == ellipap:
    #     if rs is None or rp is None:
    #         raise ValueError("Both rp and rs must be provided to design an "
    #                          "elliptic filter.")
    #     z, p, k = typefunc(N, rp, rs)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % ftype)

    # Pre-warp frequencies for digital filter design
    if not analog:
        if numpy.any(Wn <= 0) or numpy.any(Wn >= 1):
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < Wn < 1")
        fs = 2.0
        warped = 2 * fs * tan(pi * Wn / fs)
    else:
        warped = Wn

    # transform to lowpass, bandpass, highpass, or bandstop
    if btype in ('lowpass', 'highpass'):
        if numpy.size(Wn) != 1:
            raise ValueError('Must specify a single critical frequency Wn')

        if btype == 'lowpass':
            z, p, k = _zpklp2lp(z, p, k, wo=warped)
        elif btype == 'highpass':
            z, p, k = _zpklp2hp(z, p, k, wo=warped)
    elif btype in ('bandpass', 'bandstop'):
        try:
            bw = warped[1] - warped[0]
            wo = sqrt(warped[0] * warped[1])
        except IndexError:
            raise ValueError('Wn must specify start and stop frequencies')

        if btype == 'bandpass':
            z, p, k = _zpklp2bp(z, p, k, wo=wo, bw=bw)
        elif btype == 'bandstop':
            z, p, k = _zpklp2bs(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % btype)

    # Find discrete equivalent if necessary
    if not analog:
        z, p, k = _zpkbilinear(z, p, k, fs=fs)

    # Transform to proper out type (pole-zero, state-space, numer-denom)
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
        return zpk2tf(z, p, k)

def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

    The filter will have an angular (e.g. rad/s) cutoff frequency of 1.

    See Also
    --------
    butter : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = numpy.array([])
    m = numpy.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -numpy.exp(1j * pi * m / (2 * N))
    k = 1
    return z, p, k

def cheb1ap(N, rp):
    """
    Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

    The returned filter prototype has `rp` decibels of ripple in the passband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    cheby1 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero error
        # Even order filters have DC gain of -rp dB
        return numpy.array([]), numpy.array([]), 10**(-rp/20)
    z = numpy.array([])

    # Ripple factor (epsilon)
    eps = numpy.sqrt(10 ** (0.1 * rp) - 1.0)
    mu = 1.0 / N * arcsinh(1 / eps)

    # Arrange poles in an ellipse on the left half of the S-plane
    m = numpy.arange(-N+1, N, 2)
    theta = pi * m / (2*N)
    p = -sinh(mu + 1j*theta)

    k = numpy.prod(-p, axis=0).real
    if N % 2 == 0:
        k = k / sqrt((1 + eps * eps))

    return z, p, k

def cheb2ap(N, rs):
    """
    Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

    The returned filter prototype has `rs` decibels of ripple in the stopband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first reaches ``-rs``.

    See Also
    --------
    cheby2 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero warning
        return numpy.array([]), numpy.array([]), 1

    # Ripple factor (epsilon)
    de = 1.0 / sqrt(10 ** (0.1 * rs) - 1)
    mu = arcsinh(1.0 / de) / N

    if N % 2:
        m = numpy.concatenate((numpy.arange(-N+1, 0, 2),
                               numpy.arange(2, N, 2)))
    else:
        m = numpy.arange(-N+1, N, 2)

    z = -conjugate(1j / sin(m * pi / (2.0 * N)))

    # Poles around the unit circle like Butterworth
    p = -exp(1j * pi * numpy.arange(-N+1, N, 2) / (2 * N))
    # Warp into Chebyshev II
    p = sinh(mu) * p.real + 1j * cosh(mu) * p.imag
    p = 1.0 / p

    k = (numpy.prod(-p, axis=0) / numpy.prod(-z, axis=0)).real
    return z, p, k

def _zpkbilinear(z, p, k, fs):
    """
    Return a digital filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    z : array_like
        Zeros of the analog IIR filter transfer function.
    p : array_like
        Poles of the analog IIR filter transfer function.
    k : float
        System gain of the analog IIR filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g. hertz). No prewarping is
        done in this function.

    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.

    """
    z = atleast_1d(z)
    p = atleast_1d(p)

    degree = _relative_degree(z, p)

    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = append(z_z, -ones(degree))

    # Compensate for gain change
    k_z = k * real(prod(fs2 - z) / prod(fs2 - p))

    return z_z, p_z, k_z

def _zpklp2lp(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog IIR filter transfer function.
    p : array_like
        Poles of the analog IIR filter transfer function.
    k : float
        System gain of the analog IIR filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g. rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed low-pass filter transfer function.
    p : ndarray
        Poles of the transformed low-pass filter transfer function.
    k : float
        System gain of the transformed low-pass filter.

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s}{\omega_0}

    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)  # Avoid int wraparound

    degree = _relative_degree(z, p)

    # Scale all points radially from origin to shift cutoff frequency
    z_lp = wo * z
    p_lp = wo * p

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    k_lp = k * wo**degree

    return z_lp, p_lp, k_lp

def _zpklp2hp(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog IIR filter transfer function.
    p : array_like
        Poles of the analog IIR filter transfer function.
    k : float
        System gain of the analog IIR filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g. rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed high-pass filter transfer function.
    p : ndarray
        Poles of the transformed high-pass filter transfer function.
    k : float
        System gain of the transformed high-pass filter.

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.

    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)

    degree = _relative_degree(z, p)

    # Invert positions radially about unit circle to convert LPF to HPF
    # Scale all points radially from origin to shift cutoff frequency
    z_hp = wo / z
    p_hp = wo / p

    # If lowpass had zeros at infinity, inverting moves them to origin.
    z_hp = append(z_hp, zeros(degree))

    # Cancel out gain change caused by inversion
    k_hp = k * real(prod(-z) / prod(-p))

    return z_hp, p_hp, k_hp

def _zpklp2bp(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog IIR filter transfer function.
    p : array_like
        Poles of the analog IIR filter transfer function.
    k : float
        System gain of the analog IIR filter transfer function.
    wo : float
        Desired passband center, as angular frequency (e.g. rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g. rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-pass filter transfer function.
    p : ndarray
        Poles of the transformed band-pass filter transfer function.
    k : float
        System gain of the transformed band-pass filter.

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}

    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.

    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw/2
    p_lp = p * bw/2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = concatenate((z_lp + sqrt(z_lp**2 - wo**2),
                        z_lp - sqrt(z_lp**2 - wo**2)))
    p_bp = concatenate((p_lp + sqrt(p_lp**2 - wo**2),
                        p_lp - sqrt(p_lp**2 - wo**2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = append(z_bp, zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp

def _zpklp2bs(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    stopband width `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog IIR filter transfer function.
    p : array_like
        Poles of the analog IIR filter transfer function.
    k : float
        System gain of the analog IIR filter transfer function.
    wo : float
        Desired stopband center, as angular frequency (e.g. rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g. rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-stop filter transfer function.
    p : ndarray
        Poles of the transformed band-stop filter transfer function.
    k : float
        System gain of the transformed band-stop filter.

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.

    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Invert to a highpass filter with desired bandwidth
    z_hp = (bw/2) / z
    p_hp = (bw/2) / p

    # Square root needs to produce complex result, not NaN
    z_hp = z_hp.astype(complex)
    p_hp = p_hp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bs = concatenate((z_hp + sqrt(z_hp**2 - wo**2),
                        z_hp - sqrt(z_hp**2 - wo**2)))
    p_bs = concatenate((p_hp + sqrt(p_hp**2 - wo**2),
                        p_hp - sqrt(p_hp**2 - wo**2)))

    # Move any zeros that were at infinity to the center of the stopband
    z_bs = append(z_bs, +1j*wo * ones(degree))
    z_bs = append(z_bs, -1j*wo * ones(degree))

    # Cancel out gain change caused by inversion
    k_bs = k * real(prod(-z) / prod(-p))

    return z_bs, p_bs, k_bs

def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree

def cheb1ord(wp, ws, gpass, gstop, analog=False):
    """Chebyshev type I filter order selection.

    Return the order of the lowest order digital or analog Chebyshev Type I
    filter that loses no more than `gpass` dB in the passband and has at
    least `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.
        For digital filters, these are normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g. rad/s).

    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.

    Returns
    -------
    ord : int
        The lowest order for a Chebyshev type I filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `cheby1` to give filter results.

    See Also
    --------
    cheby1 : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design a digital lowpass filter such that the passband is within 3 dB up
    to 0.2*(fs/2), while rejecting at least -40 dB above 0.3*(fs/2).  Plot its
    frequency response, showing the passband and stopband constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> N, Wn = signal.cheb1ord(0.2, 0.3, 3, 40)
    >>> b, a = signal.cheby1(N, 3, Wn, 'low')
    >>> w, h = signal.freqz(b, a)
    >>> plt.semilogx(w / np.pi, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev I lowpass filter fit to constraints')
    >>> plt.xlabel('Normalized frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([.01, 0.2, 0.2, .01], [-3, -3, -99, -99], '0.9', lw=0) # stop
    >>> plt.fill([0.3, 0.3,   2,   2], [ 9, -40, -40,  9], '0.9', lw=0) # pass
    >>> plt.axis([0.08, 1, -60, 3])
    >>> plt.show()

    """
    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    filter_type = 2 * (len(wp) - 1)
    if wp[0] < ws[0]:
        filter_type += 1
    else:
        filter_type += 2

    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = tan(pi * wp / 2.0)
        stopb = tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0

    if filter_type == 1:           # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:     # stop
        wp0 = fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop, 'cheby'),
                                 disp=0)
        passb[0] = wp0
        wp1 = fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop, 'cheby'),
                                 disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:  # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))

    nat = min(abs(nat))

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    ord = int(ceil(arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))) /
                   arccosh(nat)))

    # Natural frequencies are just the passband edges
    if not analog:
        wn = (2.0 / pi) * arctan(passb)
    else:
        wn = passb

    if len(wn) == 1:
        wn = wn[0]
    return ord, wn

def cheb2ord(wp, ws, gpass, gstop, analog=False):
    """Chebyshev type II filter order selection.

    Return the order of the lowest order digital or analog Chebyshev Type II
    filter that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.
        For digital filters, these are normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g. rad/s).

    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.

    Returns
    -------
    ord : int
        The lowest order for a Chebyshev type II filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `cheby2` to give filter results.

    See Also
    --------
    cheby2 : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design a digital bandstop filter which rejects -60 dB from 0.2*(fs/2) to
    0.5*(fs/2), while staying within 3 dB below 0.1*(fs/2) or above
    0.6*(fs/2).  Plot its frequency response, showing the passband and
    stopband constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> N, Wn = signal.cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
    >>> b, a = signal.cheby2(N, 60, Wn, 'stop')
    >>> w, h = signal.freqz(b, a)
    >>> plt.semilogx(w / np.pi, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev II bandstop filter fit to constraints')
    >>> plt.xlabel('Normalized frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([.01, .1, .1, .01], [-3,  -3, -99, -99], '0.9', lw=0) # stop
    >>> plt.fill([.2,  .2, .5,  .5], [ 9, -60, -60,   9], '0.9', lw=0) # pass
    >>> plt.fill([.6,  .6,  2,   2], [-99, -3,  -3, -99], '0.9', lw=0) # stop
    >>> plt.axis([0.06, 1, -80, 3])
    >>> plt.show()

    """
    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    filter_type = 2 * (len(wp) - 1)
    if wp[0] < ws[0]:
        filter_type += 1
    else:
        filter_type += 2

    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = tan(pi * wp / 2.0)
        stopb = tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0

    if filter_type == 1:           # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:     # stop
        wp0 = fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop, 'cheby'),
                                 disp=0)
        passb[0] = wp0
        wp1 = fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop, 'cheby'),
                                 disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:  # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))

    nat = min(abs(nat))

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    ord = int(ceil(arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))) /
                   arccosh(nat)))

    # Find frequency where analog response is -gpass dB.
    # Then convert back from low-pass prototype to the original filter.

    new_freq = cosh(1.0 / ord * arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))))
    new_freq = 1.0 / new_freq

    if filter_type == 1:
        nat = passb / new_freq
    elif filter_type == 2:
        nat = passb * new_freq
    elif filter_type == 3:
        nat = numpy.zeros(2, float)
        nat[0] = (new_freq / 2.0 * (passb[0] - passb[1]) +
                  sqrt(new_freq ** 2 * (passb[1] - passb[0]) ** 2 / 4.0 +
                       passb[1] * passb[0]))
        nat[1] = passb[1] * passb[0] / nat[0]
    elif filter_type == 4:
        nat = numpy.zeros(2, float)
        nat[0] = (1.0 / (2.0 * new_freq) * (passb[0] - passb[1]) +
                  sqrt((passb[1] - passb[0]) ** 2 / (4.0 * new_freq ** 2) +
                       passb[1] * passb[0]))
        nat[1] = passb[0] * passb[1] / nat[0]

    if not analog:
        wn = (2.0 / pi) * arctan(nat)
    else:
        wn = nat

    if len(wn) == 1:
        wn = wn[0]
    return ord, wn

def buttord(wp, ws, gpass, gstop, analog=False):
    """Butterworth filter order selection.

    Return the order of the lowest order digital or analog Butterworth filter
    that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.
        For digital filters, these are normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g. rad/s).

    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.

    Returns
    -------
    ord : int
        The lowest order for a Butterworth filter which meets specs.
    wn : ndarray or float
        The Butterworth natural frequency (i.e. the "3dB frequency").  Should
        be used with `butter` to give filter results.

    See Also
    --------
    butter : Filter design using order and critical points
    cheb1ord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design an analog bandpass filter with passband within 3 dB from 20 to
    50 rad/s, while rejecting at least -40 dB below 14 and above 60 rad/s.
    Plot its frequency response, showing the passband and stopband
    constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> N, Wn = signal.buttord([20, 50], [14, 60], 3, 40, True)
    >>> b, a = signal.butter(N, Wn, 'band', True)
    >>> w, h = signal.freqs(b, a, np.logspace(1, 2, 500))
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth bandpass filter fit to constraints')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([1,  14,  14,   1], [-40, -40, 99, 99], '0.9', lw=0) # stop
    >>> plt.fill([20, 20,  50,  50], [-99, -3, -3, -99], '0.9', lw=0) # pass
    >>> plt.fill([60, 60, 1e9, 1e9], [99, -40, -40, 99], '0.9', lw=0) # stop
    >>> plt.axis([10, 100, -60, 3])
    >>> plt.show()

    """
    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    filter_type = 2 * (len(wp) - 1)
    filter_type += 1
    if wp[0] >= ws[0]:
        filter_type += 1

    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = tan(pi * wp / 2.0)
        stopb = tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0

    if filter_type == 1:            # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:          # stop
        wp0 = fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop,
                                       'butter'),
                                 disp=0)
        passb[0] = wp0
        wp1 = fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop,
                                       'butter'),
                                 disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:          # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))

    nat = min(abs(nat))

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    ord = int(ceil(log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat))))

    # Find the Butterworth natural frequency WN (or the "3dB" frequency")
    # to give exactly gpass at passb.
    try:
        W0 = (GPASS - 1.0) ** (-1.0 / (2.0 * ord))
    except ZeroDivisionError:
        W0 = 1.0
        print("Warning, order is zero...check input parameters.")

    # now convert this frequency back from lowpass prototype
    # to the original analog filter

    if filter_type == 1:  # low
        WN = W0 * passb
    elif filter_type == 2:  # high
        WN = passb / W0
    elif filter_type == 3:  # stop
        WN = numpy.zeros(2, float)
        discr = sqrt((passb[1] - passb[0]) ** 2 +
                     4 * W0 ** 2 * passb[0] * passb[1])
        WN[0] = ((passb[1] - passb[0]) + discr) / (2 * W0)
        WN[1] = ((passb[1] - passb[0]) - discr) / (2 * W0)
        WN = numpy.sort(abs(WN))
    elif filter_type == 4:  # pass
        W0 = numpy.array([-W0, W0], float)
        WN = (-W0 * (passb[1] - passb[0]) / 2.0 +
              sqrt(W0 ** 2 / 4.0 * (passb[1] - passb[0]) ** 2 +
                   passb[0] * passb[1]))
        WN = numpy.sort(abs(WN))
    else:
        raise ValueError("Bad type: %s" % filter_type)

    if not analog:
        wn = (2.0 / pi) * arctan(WN)
    else:
        wn = WN

    if len(wn) == 1:
        wn = wn[0]
    return ord, wn

def butter(N, Wn, btype='low', analog=False, output='ba'):
    """
    Butterworth digital and analog filter design.

    Design an Nth-order digital or analog Butterworth filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For a Butterworth filter, this is the point at which the gain
        drops to 1/sqrt(2) that of the passband (the "-3 dB point").
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    buttord, buttap

    Notes
    -----
    The Butterworth filter has maximally flat frequency response in the
    passband.

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Plot the filter's frequency response, showing the critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.butter(4, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth filter frequency response')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.show()

    """
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='butter')

def cheby1(N, rp, Wn, btype='low', analog=False, output='ba'):
    """
    Chebyshev type I digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type I filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rp : float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For Type I filters, this is the point in the transition band at which
        the gain first drops below -`rp`.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    cheb1ord, cheb1ap

    Notes
    -----
    The Chebyshev type I filter maximizes the rate of cutoff between the
    frequency response's passband and stopband, at the expense of ripple in
    the passband and increased ringing in the step response.

    Type I filters roll off faster than Type II (`cheby2`), but Type II
    filters do not have any ripple in the passband.

    The equiripple passband has N maxima or minima (for example, a
    5th-order filter has 3 maxima and 2 minima).  Consequently, the DC gain is
    unity for odd-order filters, or -rp dB for even-order filters.

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Plot the filter's frequency response, showing the critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.cheby1(4, 5, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev Type I frequency response (rp=5)')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.axhline(-5, color='green') # rp
    >>> plt.show()

    """
    return iirfilter(N, Wn, rp=rp, btype=btype, analog=analog,
                     output=output, ftype='cheby1')

def cheby2(N, rs, Wn, btype='low', analog=False, output='ba'):
    """
    Chebyshev type II digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type II filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rs : float
        The minimum attenuation required in the stop band.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For Type II filters, this is the point in the transition band at which
        the gain first reaches -`rs`.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    cheb2ord, cheb2ap

    Notes
    -----
    The Chebyshev type II filter maximizes the rate of cutoff between the
    frequency response's passband and stopband, at the expense of ripple in
    the stopband and increased ringing in the step response.

    Type II filters do not roll off as fast as Type I (`cheby1`).

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Plot the filter's frequency response, showing the critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.cheby2(4, 40, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev Type II frequency response (rs=40)')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.axhline(-40, color='green') # rs
    >>> plt.show()

    """
    return iirfilter(N, Wn, rs=rs, btype=btype, analog=analog,
                     output=output, ftype='cheby2')

def band_stop_obj(wp, ind, passb, stopb, gpass, gstop, type):
    """
    Band Stop Objective Function for order minimization.

    Returns the non-integer order for an analog band stop filter.

    Parameters
    ----------
    wp : scalar
        Edge of passband `passb`.
    ind : int, {0, 1}
        Index specifying which `passb` edge to vary (0 or 1).
    passb : ndarray
        Two element sequence of fixed passband edges.
    stopb : ndarray
        Two element sequence of fixed stopband edges.
    gstop : float
        Amount of attenuation in stopband in dB.
    gpass : float
        Amount of ripple in the passband in dB.
    type : {'butter', 'cheby', 'ellip'}
        Type of filter.

    Returns
    -------
    n : scalar
        Filter order (possibly non-integer).

    """
    passbC = passb.copy()
    passbC[ind] = wp
    nat = (stopb * (passbC[0] - passbC[1]) /
           (stopb ** 2 - passbC[0] * passbC[1]))
    nat = min(abs(nat))

    if type == 'butter':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        n = (log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat)))
    elif type == 'cheby':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        n = arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))) / arccosh(nat)
    elif type == 'ellip':
        GSTOP = 10 ** (0.1 * gstop)
        GPASS = 10 ** (0.1 * gpass)
        arg1 = sqrt((GPASS - 1.0) / (GSTOP - 1.0))
        arg0 = 1.0 / nat
        d0 = ellipk([arg0 ** 2, 1 - arg0 ** 2])
        d1 = ellipk([arg1 ** 2, 1 - arg1 ** 2])
        n = (d0[0] * d1[1] / (d0[1] * d1[0]))
    else:
        raise ValueError("Incorrect type: %s" % type)
    return n


band_dict = {'band': 'bandpass',
             'bandpass': 'bandpass',
             'pass': 'bandpass',
             'bp': 'bandpass',

             'bs': 'bandstop',
             'bandstop': 'bandstop',
             'bands': 'bandstop',
             'stop': 'bandstop',

             'l': 'lowpass',
             'low': 'lowpass',
             'lowpass': 'lowpass',
             'lp': 'lowpass',

             'high': 'highpass',
             'highpass': 'highpass',
             'h': 'highpass',
             'hp': 'highpass',
             }

filter_dict = {'butter': [buttap],
               'butterworth': [buttap],

               # 'cauer': [ellipap, ellipord],
               # 'elliptic': [ellipap, ellipord],
               # 'ellip': [ellipap, ellipord],

               # 'bessel': [besselap],
               # 'bessel_phase': [besselap],
               # 'bessel_delay': [besselap],
               # 'bessel_mag': [besselap],

               'cheby1': [cheb1ap],
               'chebyshev1': [cheb1ap],
               'chebyshevi': [cheb1ap],

               'cheby2': [cheb2ap],
               'chebyshev2': [cheb2ap],
               'chebyshevii': [cheb2ap],
               }

# filter_dict = {'butter': [buttap, buttord],
#                'butterworth': [buttap, buttord],

#                # 'cauer': [ellipap, ellipord],
#                # 'elliptic': [ellipap, ellipord],
#                # 'ellip': [ellipap, ellipord],

#                # 'bessel': [besselap],
#                # 'bessel_phase': [besselap],
#                # 'bessel_delay': [besselap],
#                # 'bessel_mag': [besselap],

#                'cheby1': [cheb1ap, cheb1ord],
#                'chebyshev1': [cheb1ap, cheb1ord],
#                'chebyshevi': [cheb1ap, cheb1ord],

#                'cheby2': [cheb2ap, cheb2ord],
#                'chebyshev2': [cheb2ap, cheb2ord],
#                'chebyshevii': [cheb2ap, cheb2ord],
#                }