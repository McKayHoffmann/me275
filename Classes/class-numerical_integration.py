# Integral -> sin(x**2) from 0 to pi

from scipy import integrate
import numpy as np
import math

f = lambda x: math.sin(x**2)

# Actual integral = 0.77265
x = np.linspace(0, math.pi, 10)
y = np.array([f(xi) for xi in x])
integrate.trapezoid(y, x)

# Convergance study
n = 10
past = 0

def convergance(n, past=0):
    while True:
        x = np.linspace(0, math.pi, n)
        y = np.array([f(xi) for xi in x])
        current = integrate.trapezoid(y, x)
        if abs(current - past) < 0.001:
            return n
        else:
            past = current
            n *= 2

x = np.linspace(0, math.pi, 10485760)
y = np.array([f(xi) for xi in x])
a = integrate.trapezoid(y, x)

x = np.linspace(0, math.pi, 20971520)
y = np.array([f(xi) for xi in x])
b = integrate.trapezoid(y, x)