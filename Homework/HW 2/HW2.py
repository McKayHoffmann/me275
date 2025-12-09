# Homework 2
# ME EN 275
# Author: McKay Hoffmann
# Date: 9-17-2025

import math
import numpy as np
import matplotlib.pyplot as plt

# -- Problem 1 -- #

def summation(n, sum=0.0):
    for _ in range(0, n):
        sum += 0.1
    return sum

def f(n):
    sum = summation(n)
    return math.fabs(n/10 - sum)

n_vec = np.array([10, 100, 1000, 10000, 100000, 1000000])
error_vec = np.array([f(n) for n in n_vec])

plt.loglog(n_vec, error_vec)
plt.ylabel("Roundoff Error")
plt.xlabel("n")
plt.title("Roundoff error vs. n")
plt.savefig("RoundoffErrorVsN.png")

# -- Problem 2 -- #

def relative_percent_error_pi(measured, true=math.pi):
    return abs((measured - true) / true) * 100

def leibniz_error(n):
    pi_sum = 0
    for i in range(0, n):
        pi_sum += ((-1)**i) / (2*i + 1)
    approximation = 4 * pi_sum
    return relative_percent_error_pi(approximation)

n_vec = np.linspace(0, 20, 21).astype(int)
percent_error_vec = np.array([leibniz_error(n) for n in n_vec])

plt.plot(n_vec, percent_error_vec)
plt.xlabel("n")
plt.ylabel("Truncation percent error (%)")
plt.xticks(n_vec)
plt.title("Truncation Percent Error vs. n")
plt.savefig("TruncationPercentErrorVsn.png")

# -- Problem 3 -- #
from scipy.optimize import root_scalar

residual = lambda theta: (78.48/ (math.cos(math.radians(theta)))**2) - ((400 * math.sin(math.radians(theta))) / math.cos(math.radians(theta))) + 50
# Somewhere between 10 and 30 degrees

def bisection(f, a, b, tol=1e-6):
    """
    Find a root of function f (i.e., f(x) = 0) within bracket [a, b] using bisection.

    Parameters
    ----------
    f : function
        the function that we are finding a root for: f(x) = 0
    a : float
        left endpoint
    b : float
        right endpoint. note f(a) * f (b) must be < 0, otherwise function will return.
    tol : float
        tolerance for stopping criteria

    Returns
    -------
    x : float
        the root where f(x) = 0
    """
    if (f(a) * f(b)) > 0:
        print("Please select a valid bracket")
        return
    while abs(a - b) > tol:
        midpoint = (a + b) / 2
        if abs(f(midpoint)) < tol:
            return midpoint
        elif f(midpoint) > 0 and f(a) > 0:
            a = midpoint
        elif f(midpoint) > 0 and f(a) < 0:
            b = midpoint
        elif f(midpoint) < 0 and f(a) < 0:
            a = midpoint
        elif f(midpoint) < 0 and f(a) > 0:
            b = midpoint

bisection(residual, 10, 30)
root_scalar(residual, bracket=(10, 30))     # Both return 19.00794506072998