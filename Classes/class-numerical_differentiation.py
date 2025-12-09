# Numerical Differentiation
# ME 275
# Author: McKay Hoffmann

# How to change directory, cd ksjdaslkdh

## Integration Review
# scipy.integrate.trapezoid (or np.trapezoid)
# scipy.integrate.simpsons
# scipy.integrate.quad

# scipy.integrate.cumulative_trapezoid
# Instead of integrating from a to b, this integrates from a to x, as x moves.
# This is helpful when you might want a graph of the integral
# that you couldn't otherwise have

# Example: sin(x)
import numpy as np
import math
from scipy import integrate
import matplotlib.pyplot as plt

x = np.linspace(0, 2*math.pi, 100_000)
y = np.array([math.sin(xi) for xi in x])

integrate.trapezoid(y, x)
integrate.simpson(y, x)
integral = integrate.cumulative_trapezoid(y, x)
integral = np.insert(integral, 0, 0)    # How to insert an element. (array, place, value)
integral = integral - 1
double_integral = integrate.cumulative_trapezoid(integral, x)
double_integral = np.insert(double_integral, 0, 0)
triple_integral = integrate.cumulative_trapezoid(double_integral, x)
triple_integral = np.insert(triple_integral, 0, 0)
triple_integral += 1

plt.plot(x, y, label="sin(x)")
plt.plot(x, integral, color="red", label="-cos(x)")
plt.plot(x, double_integral, color="green", label="-sin(x)")
plt.plot(x, triple_integral, color="magenta", label="cos(x)")
plt.legend()

f = lambda x: math.sin(x)
integrate.quad(f, 0, 1)     # Great for accuracy and performance, but you need the function
# Returns the integral and an estimate of the absolute error

## DERIVATIVES
# Finite differencing: f'(x) = (f(x+h) - f(x)) / h
# Taylor's series
# Central Differencing: Instead of evaluating one extra point, evaluate two points.
# f'(x) = (f(x+h) - f(x-h)) / 2h

# Example: f(x) = x**2
g = lambda x: x**2
h = 10**-1

def finite_differencing(function, x0, h):
    return (function(x0 + h) - function(x0)) / (h)

finite_differencing(g, 1, 1e-1)
finite_differencing(g, 1, 1e-2)
finite_differencing(g, 1, 1e-3)
finite_differencing(g, 1, 1e-4)
finite_differencing(g, 1, 1e-8)
finite_differencing(g, 1, 1e-15)
finite_differencing(g, 1, 1e-16)

# h = 10e-6 is typically a good step size
# In some cases, when x is very large or very small, it can be better to take a relative step size: h = x * 10**-64

# Functions to use
import scipy
scipy.optimize.approx_fprime
np.gradient # Need to have discreet data