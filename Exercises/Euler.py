# Euler stepping exercise
# ME 275
# Author: McKay Hoffmann
# Date: 10-7-2025

import numpy as np
import matplotlib.pyplot as plt

def dydx(y, t=0):
    return 0.1 * y

def euler(func, y0, t0, tf, h):
    yAxis = np.array([y0])
    tAxis = np.array([t0])
    ti = t0
    while ti <= tf:
        yi = (func(yAxis[-1]) * h) + yAxis[-1]
        yAxis = np.append(yAxis, yi)
        ti = ti + h
        tAxis = np.append(tAxis, ti)
    return yAxis, tAxis

yAxis, tAxis = euler(dydx, 1, 0, 100, 10e-4)

def realFunction(t):
    return np.exp(0.1*t)

tReal = np.linspace(0, 100, 100_000)
yReal = np.array([realFunction(ti) for ti in tReal])

plt.plot(tAxis, yAxis, label="Approximation")
plt.plot(tReal, yReal, "-", label="Real function", color="red")

def stabilityExample(y, t=0):
    return -y

yAxis, tAxis = euler(stabilityExample, 5, 0, 10, 2)