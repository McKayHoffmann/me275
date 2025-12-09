# HW 4
# ME 275
# Author: McKay Hoffmann
# Data: 10-1-2025

# -- Problem 1 -- #
import numpy as np
from scipy.optimize import root_scalar
from scipy import integrate
import math

def getIntegrationEquation(r):
    def integrationEquation(x):
        return (math.sqrt(r**2 - x**2)) + (math.sqrt(1 - x**2) - 1)
    return integrationEquation

def findGoatArea(r):
    lowerBound = -math.sqrt(r**2 - ((r**4) / (4)))
    upperBound = lowerBound * -1
    integrationEquation = getIntegrationEquation(r)
    return integrate.quad(integrationEquation, lowerBound, upperBound)[0]

findGoatArea(1)     # Returns 1.228369698608755

residual = lambda r: findGoatArea(r) - math.pi/2

root = root_scalar(residual, x0=1).root
# root = 1.158728473018125
residual(root)

# -- Problem 2 -- #
f = lambda x: math.exp(x) * math.sin(x)
xstar = 1.5
fPrime = lambda x: (math.exp(x) * math.sin(x)) + (math.exp(x) * math.cos(x))
exactAnswer = fPrime(xstar)

def forwardFiniteDifferencing(function, x0, h):
    return (function(x0 + h) - function(x0)) / (h)

def percent_error(true, measured):
    return abs((true - measured) / true) * 100

estimate = forwardFiniteDifferencing(f, xstar, 10e-1)
percent_error(exactAnswer, estimate)    # 41.08762360261747%
estimate = forwardFiniteDifferencing(f, xstar, 10e-6)
percent_error(exactAnswer, estimate)    # 6.621636054316015e-05%
estimate = forwardFiniteDifferencing(f, xstar, 10e-11)
percent_error(exactAnswer, estimate)    # 3.008802119708169e-05%
estimate = forwardFiniteDifferencing(f, xstar, 10e-17)
percent_error(exactAnswer, estimate)    # 100.0%
# Percent error drops down with a smaller step size, and then goes back up
# Subtractive cancelation error

# -- Problem 3 -- #
f = lambda x: math.exp(x) * math.sin(x)

def myTrapezoid(f, steps):
    x = np.linspace(0, 1, steps)
    y = np.array([f(xi) for xi in x])
    result = 0
    for i in range(0, len(x)-1):
        slice = ((y[i] + y[i+1]) / 2) * (x[i+1] - x[i])
        result += slice
    return result

def convergance(f, n, past=0):
    while True:
        current = myTrapezoid(f, n)
        if abs(current - past) < 10e-10:
            return n, current
        else:
            past = current
            n += 10

n, current = convergance(f, 10)
n       # 180
current     # 0.909337841663893
actualIntegral = integrate.quad(f, 0, 1)[0]     # 0.9093306736314788

# -- Problem 4 -- #
import matplotlib.pyplot as plt
import numpy as np
time = np.array([0.0, 0.24489796, 0.44897959, 0.65306122, 0.85714286, 1.06122449, 1.26530612, 1.46938776, 1.67346939, 1.87755102, 2.08163265, 2.28571429, 2.48979592, 2.69387755, 2.89795918, 3.10204082, 3.30612245, 3.51020408, 3.71428571, 3.91836735, 4.12244898, 4.32653061, 4.53061224, 4.73469388, 4.93877551, 5.14285714, 5.34693878, 5.55102041, 5.75510204, 5.95918367, 6.16326531, 6.36734694, 6.57142857, 6.7755102 , 6.97959184, 7.18367347, 7.3877551 , 7.59183673, 7.79591837, 8.])
position = np.array([0.16326531, 0.97959184, 1.79591837, 2.6122449,  3.42857143, 4.24489796, 5.06122449, 5.87755102, 6.69387755, 7.51020408, 8.32653061, 9.14285714, 9.95918367, 10.7755102, 11.59183673, 12.40816327, 13.2244898, 14.04081633, 14.85714286, 15.67346939, 16.46730529, 17.14618909, 17.70012495, 18.12911287, 18.43315285, 18.6122449, 18.666389, 18.59558517, 18.3998334, 18.07913369, 17.63348605, 17.06289046, 16.36734694, 15.54685548, 14.60141608, 13.53102874, 12.33569346, 11.01541025, 9.57017909, 8.])
velocity = np.gradient(position, time)

plt.plot(time, position, label="Position (m)")
plt.plot(time, velocity, label="Velocity (m/s)")
plt.xlabel("Time (s)")
plt.legend()
plt.savefig("Numerically differentiaing velocity.png")
