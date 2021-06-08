"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener
process).

Adapted from scipy with the addition of a drift parameter (mean of Gaussian
can be non-zero).
"""

import numpy as np


def brownian(x0, n, dt, delta, drift=0, out=None, rng=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(drift, delta**2 * t; 0, t)

    where N(a, b; t0, t1) is a normally distributed random variable with mean a
    and variance b. The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(drift, delta**2 * dt; t, t+dt)

    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy
         array using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random
        variable of the position at time t, X(t), has a normal distribution
        whose mean is the position at time t=0 and whose variance is
        delta**2*t.
    drift : float or numpy array.
        Mean value of random walk.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.
    rng : np.random.default_rng()
        Random number generator.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    if rng is None:
        rng = np.random.default_rng()

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = rng.normal(loc=drift, scale=delta * np.sqrt(dt), size=(n,) + x0.shape)

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=0, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=0)

    return out
