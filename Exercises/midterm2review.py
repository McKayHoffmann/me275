# First-Order ODE
import numpy as np

def dydx(t, y):
    return y - t**2 + 1

def myEuler(func, y0, t0, tf, h):
    yAxis = np.array([y0])
    tAxis = np.array([t0])
    ti = t0
    while ti <= tf:
        yi = (func(yAxis[-1], ti) * h) + yAxis[-1]
        yAxis = np.append(yAxis, yi)
        ti = ti + h
        tAxis = np.append(tAxis, ti)
    return yAxis, tAxis

yAxis, tAxis = myEuler(dydx, 0.5, 0, 0.2, 10e-4)
yAxis, tAxis = myEuler(dydx, 0.5, 0, 0.2, 10e-5)
yAxis, tAxis = myEuler(dydx, 0.5, 0, 0.2, 10e-6)
# y(0.2) = 0.829


def dydx(t, y):
    return y - t**2 + 1

def myEuler2(func, y0, t_range, step_size):
    ti, tf = t_range
    h = step_size
    y_axis = np.array([y0])
    t_axis = np.

# System of First-Order ODEs
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system(t, y):
    x, y = y
    xdot = y
    ydot = -x
    return xdot, ydot

my_time = np.arange(0, 3, 1e-6)
sol = solve_ivp(system, (0,3), [1, 0], t_eval=my_time)

sol.y
# Trajectory
plt.plot(sol.y[0], sol.y[1], '->')
plt.xlabel("x"); plt.ylabel("y")

# Phase plot
plt.plot(my_time, sol.y[0, :], label="x")
plt.plot(my_time, sol.y[1, :], label="y")
plt.legend()

# Higher-Order ODE
def oscillator(t, y):
    u, udot = y
    udot = udot
    uddot = -2*udot -5*u
    return udot, uddot

my_time1 = np.arange(0, 10, 1e-6)
sol1 = solve_ivp(oscillator, (0, 10), [1, 0], t_eval=my_time1)
plt.plot(sol1.t, sol1.y[0, :])

# Coupled Second-Order System
def mass_spring(t, y):
    x1, x2, x1dot, x2dot = y
    x1ddot = -2*(x1 - x2)
    x2ddot = 2*(x1 - x2)
    return x1dot, x2dot, x1ddot, x2ddot

my_time3 = np.arange(0, 4, 1e-3)
sol2 = solve_ivp(mass_spring, (0, 4), [1, 0, 0, 0], t_eval=my_time3)
plt.plot(sol2.t, sol2.y[0, :], label="x1")
plt.plot(sol2.t, sol2.y[1, :], '-', label="x2")
plt.legend()

# Correlation and Linear Regression
import scipy.stats as stats
hours = np.array([2, 4, 6, 8, 10])
score = np.array([65, 70, 75, 85, 95])
plt.scatter(hours, score)
# a)
r = stats.pearsonr(hours, score)[0]
# b)
A = np.column_stack((hours, np.ones_like(hours)))
a, b = np.linalg.lstsq(A, score)[0]

y_vec = np.array([(a*xi + b) for xi in hours])
plt.scatter(hours, score)
plt.plot(hours, y_vec)
