from __future__ import division, print_function, absolute_import

import warnings
import numpy
from numpy import (atleast_1d, eye, mgrid, argmin, zeros, shape, squeeze,
                   vectorize, asarray, sqrt, Inf, asfarray, isinf, finfo, 
                   sign, sqrt)
import numpy as np
import warnings

__all__ = ['newton', 'fmin', 'fbound']


# Newton-Raphson method
def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None):
    """
    Find a zero using the Newton-Raphson or secant method.

    Find a zero of the function `func` given a nearby starting point `x0`.
    The Newton-Raphson method is used if the derivative `fprime` of `func`
    is provided, otherwise the secant method is used.  If the second order
    derivate `fprime2` of `func` is provided, parabolic Halley's method
    is used.

    Parameters
    ----------
    func : function
        The function whose zero is wanted. It must be a function of a
        single variable of the form f(x,a,b,c...), where a,b,c... are extra
        arguments that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    fprime : function, optional
        The derivative of the function when available and convenient. If it
        is None (default), then the secant method is used.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value.
    maxiter : int, optional
        Maximum number of iterations.
    fprime2 : function, optional
        The second order derivative of the function when available and
        convenient. If it is None (default), then the normal Newton-Raphson
        or the secant method is used. If it is given, parabolic Halley's
        method is used.

    Returns
    -------
    zero : float
        Estimated location where function is zero.

    See Also
    --------
    brentq, brenth, ridder, bisect
    fsolve : find zeroes in n dimensions.

    Notes
    -----
    The convergence rate of the Newton-Raphson method is quadratic,
    the Halley method is cubic, and the secant method is
    sub-quadratic.  This means that if the function is well behaved
    the actual error in the estimated zero is approximately the square
    (cube for Halley) of the requested tolerance up to roundoff
    error. However, the stopping criterion used here is the step size
    and there is no guarantee that a zero has been found. Consequently
    the result should be verified. Safer algorithms are brentq,
    brenth, ridder, and bisect, but they all require that the root
    first be bracketed in an interval where the function changes
    sign. The brentq algorithm is recommended for general use in one
    dimensional problems when such an interval has been found.

    Examples
    --------

    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1
    
    >>> from scipy import optimize

    ``fprime`` and ``fprime2`` not provided, use secant method
    
    >>> root = optimize.newton(f, 1.5)
    >>> root
    1.0000000000000016

    Only ``fprime`` provided, use Newton Raphson method
    
    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2)
    >>> root
    1.0
    
    ``fprime2`` provided, ``fprime`` provided/not provided use parabolic
    Halley's method

    >>> root = optimize.newton(f, 1.5, fprime2=lambda x: 6 * x)
    >>> root
    1.0000000000000016
    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2,
    ...                        fprime2=lambda x: 6 * x)
    >>> root
    1.0

    """
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if fprime is not None:
        # Newton-Rapheson method
        # Multiply by 1.0 to convert to floating point.  We don't use float(x0)
        # so it still works if x0 is complex.
        p0 = 1.0 * x0
        fder2 = 0
        for iter in range(maxiter):
            myargs = (p0,) + args
            fder = fprime(*myargs)
            if fder == 0:
                msg = "derivative was zero."
                warnings.warn(msg, RuntimeWarning)
                return p0
            fval = func(*myargs)
            if fprime2 is not None:
                fder2 = fprime2(*myargs)
            if fder2 == 0:
                # Newton step
                p = p0 - fval / fder
            else:
                # Parabolic Halley's method
                discr = fder ** 2 - 2 * fval * fder2
                if discr < 0:
                    p = p0 - fder / fder2
                else:
                    p = p0 - 2*fval / (fder + sign(fder) * sqrt(discr))
            if abs(p - p0) < tol:
                return p
            p0 = p
    else:
        # Secant method
        p0 = x0
        if x0 >= 0:
            p1 = x0*(1 + 1e-4) + 1e-4
        else:
            p1 = x0*(1 + 1e-4) - 1e-4
        q0 = func(*((p0,) + args))
        q1 = func(*((p1,) + args))
        for iter in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = "Tolerance of %s reached" % (p1 - p0)
                    warnings.warn(msg, RuntimeWarning)
                return (p1 + p0)/2.0
            else:
                p = p1 - q1*(p1 - p0)/(q1 - q0)
            if abs(p - p1) < tol:
                return p
            p0 = p1
            q0 = q1
            p1 = p
            q1 = func(*((p1,) + args))
    msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
    raise RuntimeError(msg)

def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None, initial_simplex=None):
    """
    Minimize a function using the downhill simplex algorithm.

    This algorithm only uses function values, not derivatives or second
    derivatives.

    Parameters
    ----------
    func : callable func(x,*args)
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func, i.e. ``f(x,*args)``.
    xtol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    ftol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : number, optional
        Maximum number of function evaluations to make.
    full_output : bool, optional
        Set to True if fopt and warnflag outputs are desired.
    disp : bool, optional
        Set to True to print convergence messages.
    retall : bool, optional
        Set to True to return list of solutions at each iteration.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    initial_simplex : array_like of shape (N + 1, N), optional
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the j-th vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.

    Returns
    -------
    xopt : ndarray
        Parameter that minimizes function.
    fopt : float
        Value of function at minimum: ``fopt = func(xopt)``.
    iter : int
        Number of iterations performed.
    funcalls : int
        Number of function calls made.
    warnflag : int
        1 : Maximum number of function evaluations made.
        2 : Maximum number of iterations reached.
    allvecs : list
        Solution at each iteration.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'Nelder-Mead' `method` in particular.

    Notes
    -----
    Uses a Nelder-Mead simplex algorithm to find the minimum of function of
    one or more variables.

    This algorithm has a long history of successful use in applications.
    But it will usually be slower than an algorithm that uses first or
    second derivative information. In practice it can have poor
    performance in high-dimensional problems and is not robust to
    minimizing complicated functions. Additionally, there currently is no
    complete theory describing when the algorithm will successfully
    converge to the minimum, or how fast it will if it does. Both the ftol and
    xtol criteria must be met for convergence.

    Examples
    --------
    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fmin(f, 1)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 17
             Function evaluations: 34
    >>> minimum[0]
    -8.8817841970012523e-16

    References
    ----------
    .. [1] Nelder, J.A. and Mead, R. (1965), "A simplex method for function
           minimization", The Computer Journal, 7, pp. 308-313

    .. [2] Wright, M.H. (1996), "Direct Search Methods: Once Scorned, Now
           Respectable", in Numerical Analysis 1995, Proceedings of the
           1995 Dundee Biennial Conference in Numerical Analysis, D.F.
           Griffiths and G.A. Watson (Eds.), Addison Wesley Longman,
           Harlow, UK, pp. 191-208.

    """
    opts = {'xatol': xtol,
            'fatol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'return_all': retall,
            'initial_simplex': initial_simplex}

    res = _minimize_neldermead(func, x0, args, callback=callback, **opts)
    if full_output:
        retlist = res['x'], res['fun'], res['nit'], res['nfev'], res['status']
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_neldermead(func, x0, args=(), callback=None,
                         maxiter=None, maxfev=None, disp=False,
                         return_all=False, initial_simplex=None,
                         xatol=1e-4, fatol=1e-4, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the j-th vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
    xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.

    """
    if 'ftol' in unknown_options:
        warnings.warn("ftol is deprecated for Nelder-Mead,"
                      " use fatol instead. If you specified both, only"
                      " fatol is used.",
                      DeprecationWarning)
        if (np.isclose(fatol, 1e-4) and
                not np.isclose(unknown_options['ftol'], 1e-4)):
            # only ftol was probably specified, use it.
            fatol = unknown_options['ftol']
        unknown_options.pop('ftol')
    if 'xtol' in unknown_options:
        warnings.warn("xtol is deprecated for Nelder-Mead,"
                      " use xatol instead. If you specified both, only"
                      " xatol is used.",
                      DeprecationWarning)
        if (np.isclose(xatol, 1e-4) and
                not np.isclose(unknown_options['xtol'], 1e-4)):
            # only xtol was probably specified, use it.
            xatol = unknown_options['xtol']
        unknown_options.pop('xtol')

    _check_unknown_options(unknown_options)
    maxfun = maxfev
    retall = return_all

    fcalls, func = wrap_function(func, args)

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025

    x0 = asfarray(x0).flatten()

    if initial_simplex is None:
        N = len(x0)

        sim = numpy.zeros((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = numpy.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.asfarray(initial_simplex).copy()
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    if retall:
        allvecs = [sim[0]]

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf

    one2np1 = list(range(1, N + 1))
    fsim = numpy.zeros((N + 1,), float)

    for k in range(N + 1):
        fsim[k] = func(sim[k])

    ind = numpy.argsort(fsim)
    fsim = numpy.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = numpy.take(sim, ind, 0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        if (numpy.max(numpy.ravel(numpy.abs(sim[1:] - sim[0]))) <= xatol and
                numpy.max(numpy.abs(fsim[0] - fsim[1:])) <= fatol):
            break

        xbar = numpy.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = numpy.argsort(fsim)
        sim = numpy.take(sim, ind, 0)
        fsim = numpy.take(fsim, ind, 0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = numpy.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print('Warning: ' + msg)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print('Warning: ' + msg)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x, final_simplex=(sim, fsim))
    if retall:
        result['allvecs'] = allvecs
    return result

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in Scipy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)

def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper


class OptimizeResult(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def fminbound(func, x1, x2, args=(), xtol=1e-5, maxfun=500,
              full_output=0, disp=1):
    """Bounded minimization for scalar functions.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized (must accept and return scalars).
    x1, x2 : float or array scalar
        The optimization bounds.
    args : tuple, optional
        Extra arguments passed to function.
    xtol : float, optional
        The convergence tolerance.
    maxfun : int, optional
        Maximum number of function evaluations allowed.
    full_output : bool, optional
        If True, return optional outputs.
    disp : int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.


    Returns
    -------
    xopt : ndarray
        Parameters (over given interval) which minimize the
        objective function.
    fval : number
        The function value at the minimum point.
    ierr : int
        An error flag (0 if converged, 1 if maximum number of
        function calls reached).
    numfunc : int
      The number of function calls made.

    See also
    --------
    minimize_scalar: Interface to minimization algorithms for scalar
        univariate functions. See the 'Bounded' `method` in particular.

    Notes
    -----
    Finds a local minimizer of the scalar function `func` in the
    interval x1 < xopt < x2 using Brent's method.  (See `brent`
    for auto-bracketing).

    Examples
    --------
    `fminbound` finds the minimum of the function in the given range.
    The following examples illustrate the same

    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fminbound(f, -1, 2)
    >>> minimum
    0.0
    >>> minimum = optimize.fminbound(f, 1, 2)
    >>> minimum
    1.0000059608609866
    """
    options = {'xatol': xtol,
               'maxiter': maxfun,
               'disp': disp}

    res = _minimize_scalar_bounded(func, (x1, x2), args, **options)
    if full_output:
        return res['x'], res['fun'], res['status'], res['nfev']
    else:
        return res['x']


def _minimize_scalar_bounded(func, bounds, args=(),
                             xatol=1e-5, maxiter=500, disp=0,
                             **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp : bool
        Set to True to print convergence messages.
    xatol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    _check_unknown_options(unknown_options)
    maxfun = maxiter
    # Test bounds are of correct form
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    x1, x2 = bounds

    if not (is_array_scalar(x1) and is_array_scalar(x2)):
        raise ValueError("Optimisation bounds must be scalars"
                         " or array scalars.")
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    flag = 0
    header = ' Func-count     x          f(x)          Procedure'
    step = '       initial'

    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * numpy.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    if disp > 2:
        print(" ")
        print(header)
        print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

    while (numpy.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if numpy.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = numpy.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((numpy.abs(p) < numpy.abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                step = '       parabolic'

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = numpy.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden section step
                golden = 1

        if golden:  # Do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e
            step = '       golden'

        si = numpy.sign(rat) + (rat == 0)
        x = xf + si * numpy.max([numpy.abs(rat), tol1])
        fu = func(x, *args)
        num += 1
        fmin_data = (num, x, fu)
        if disp > 2:
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * numpy.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    fval = fx
    if disp > 0:
        _endprint(x, flag, fval, maxfun, xatol, disp)

    result = OptimizeResult(fun=fval, status=flag, success=(flag == 0),
                            message={0: 'Solution found.',
                                     1: 'Maximum number of function calls '
                                        'reached.'}.get(flag, ''),
                            x=xf, nfev=num)

    return result


def _endprint(x, flag, fval, maxfun, xtol, disp):
    if flag == 0:
        if disp > 1:
            print("\nOptimization terminated successfully;\n"
                  "The returned value satisfies the termination criteria\n"
                  "(using xtol = ", xtol, ")")
    if flag == 1:
        if disp:
            print("\nMaximum number of function evaluations exceeded --- "
                  "increase maxfun argument.\n")
    return