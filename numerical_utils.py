# Utilities for solving DE's numerically

from collections import Sequence
import numpy as np

def getM(y0):
    """
    Computes the length of the float/int/array/list-type input y0. This
    is just glorified type checking.
    """

    if isinstance(y0, Sequence):
        return len(M)
    elif isinstance(y0, int):
        return 1
    elif isinstance(y0, float):
        return 1
    elif isinstance(y0, numpy.ndarray):
        return y0.size

def euler(f, x0, y0, h, N):
    """
    Takes N steps using the Forward Euler numerical algorithm for
    approximing the solution to the initial value problem

      y'(x) = f(x,y),       y(x0) = y0

    The stepsize is h.

    Output: array of size N+1.
    """

    assert h > 0, "Stepsize h must be positive"
    assert N >= 0, "Number of steps N must be non-negative"

    # Support for vector-valued DE's
    M = getM(y0)

    y = np.zeros([M, N+1])
    y[:,0] = y0
    x = x0

    for m in range(N):
        y[:,m+1] = y[:,m] + h*f(x, y[:,m])
        x += h

    return y

def improved_euler(f, x0, y0, h, N):
    """
    Takes N steps using the Improved Euler numerical algorithm for
    approximing the solution to the initial value problem

      y'(x) = f(x,y),       y(x0) = y0

    The stepsize is h.

    Output: array of size N+1.
    """

    assert h > 0, "Stepsize h must be positive"
    assert N >= 0, "Number of steps N must be non-negative"

    # Support for vector-valued DE's
    M = getM(y0)

    y = np.zeros([M, N+1])
    y[:,0] = y0
    x = x0

    for m in range(N):
        k1 = f(x, y[:,m])
        u = y[:,m] + h*k1
        k2 = f(x+h, u)

        y[:,m+1] = y[:,m] + h/2 * (k1 + k2)
        x += h

    return y

def runge_kutta_4(f, x0, y0, h, N):
    """
    Takes N steps using the Runge Kutta(4) numerical algorithm for
    approximing the solution to the initial value problem

      y'(x) = f(x,y),       y(x0) = y0

    The stepsize is h.

    Output: array of size N+1.
    """

    assert h > 0, "Stepsize h must be positive"
    assert N >= 0, "Number of steps N must be non-negative"

    # Support for vector-valued DE's
    M = getM(y0)

    y = np.zeros([M, N+1])
    y[:,0] = y0
    x = x0

    for m in range(N):
        k1 = f(x, y[:,m])
        k2 = f(x + h/2, y[:,m] + h/2*k1)
        k3 = f(x + h/2, y[:,m] + h/2*k2)
        k4 = f(x + h, y[:,m] + h*k3)

        y[:,m+1] = y[:,m] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        x += h

    return y
