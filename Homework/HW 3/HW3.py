# Homework 3
# ME 275
# Author: McKay Hoffmann
# Date: 9-24-2025

import math
import numpy as np

# -- Problem 1 -- #
from scipy.optimize import root_scalar

def residual(beta, theta=15, M=2):
    return ((2 / math.tan(math.radians(beta))) * (
        (M**2 * math.sin(math.radians(beta))**2 - 1)
        /
        (M**2 * (1.4 + math.cos(math.radians(2 * beta))) + 2)
    )) - math.tan(math.radians(theta))

math.degrees(math.asin(1/M))    # Returns 30 degrees
root_scalar(residual, bracket=(30, 64))     # Root = 45.343616761854406

# -- Problem 2 -- #
def findNextGuess(currentGuess, func, derivative): 
    return currentGuess - (func(currentGuess) / derivative(currentGuess))

def myNewton(func, derivative, currentGuess, tol=10e-6):
    iterations = 0
    if abs(func(currentGuess)) < tol:    # In the case the first guess is correct
        return currentGuess
    nextGuess = findNextGuess(currentGuess, func, derivative)
    deltaX = abs(nextGuess - currentGuess)
    print(deltaX)
    while deltaX > tol:
        currentGuess = nextGuess
        iterations += 1
        nextGuess = findNextGuess(currentGuess, func, derivative)
        deltaX = abs(nextGuess - currentGuess)
    return iterations, currentGuess

func = lambda x: x**3 - 3*x**2 + x - 1
derivative = lambda x: 3*x**2 - 6*x + 1

iterations, root = myNewton(func, derivative, 2.6)
iterations  # Returns 3
root    # Returns 2.769292493081928
root_scalar(func, x0=2.6)   # For verification. Returns 2.7692923542386314

# -- Problem 3 -- #
from scipy import integrate
import matplotlib.pyplot as plt
lift = lambda x: math.sqrt(1 - 4*x**2)

def convergance(f, n, past=0):
    n_log = np.array([])
    integral_log = np.array([])
    while True:
        x = np.linspace(-0.5, 0.5, n)
        y = np.array([f(xi) for xi in x])
        current = 4 * integrate.trapezoid(y, x)
        n_log = np.append(n_log, n)
        integral_log = np.append(integral_log, current)
        if abs(current - past) < 10e-10:
            return n_log, integral_log
        else:
            past = current
            n += 10

n_log, integral_log = convergance(lift, 10)
print(n_log)
print(integral_log)

quad = 4 * integrate.quad(lift, -0.5, 0.5)[0]
quad_vec = np.zeros_like(n_log) + quad

plt.plot(n_log, integral_log, label="integrate.trapezoid")
plt.xscale('log')
plt.xlabel("Discretization Points")
plt.ylabel("Integral Value (Tolerance = 10e-10)")
plt.title("Lift Distribution Convergance Study")
plt.plot(n_log, quad_vec, "--", color="red", label="integrate.quad")
plt.legend()
plt.savefig("Lift Distribution Convergance Study.png")

# -- Problem 4 -- #
data = np.loadtxt("accel.dat")
velocity = integrate.cumulative_trapezoid(data[:, 1], data[:, 0])
velocity = np.insert(velocity, 0, 0)

plt.figure()
plt.subplot(211)
plt.plot(data[:, 0], data[:, 1])
plt.ylabel("Acceleration (m/s^2)")
plt.title("First Stage")
plt.subplot(212)
plt.plot(data[:, 0], velocity)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.savefig("First Stage Data")