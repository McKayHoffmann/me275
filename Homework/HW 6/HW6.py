# HW 6
# ME 275
# Author: McKay Hoffmann
# Date: 10/23/2025
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

# -- Problem 1 -- #
def dydt(t, y):
    return y*t**2 - 1.1*y

def myEuler(fun, init_cond, h, interval):
    t_range = np.arange(interval[0], interval[1]+h, h)
    y_log = np.array([init_cond])
    for ti in t_range:
        yi1 = y_log[-1] + (fun(ti, y_log[-1]))*h
        y_log = np.append(y_log, yi1)
    return t_range, y_log[:-1]
    
interval = (0, 2)
y0 = 1
h1 = 0.5; h2 = 0.1; h3 = 0.01
t1, y1 = myEuler(dydt, y0, h1, interval)
t2, y2 = myEuler(dydt, y0, h2, interval)
t3, y3 = myEuler(dydt, y0, h3, interval)

sol1 = solve_ivp(dydt, (0, 2), [1])
t4, y4 = sol1.t, sol1.y[0]
myTime = np.arange(0, 2+1e-4, 1e-4)
sol1 = solve_ivp(dydt, (0, 2), [1], t_eval=myTime)
t5, y5 = sol1.t, sol1.y[0]

plt.figure()
plt.plot(t1, y1, label="h = 0.5")
plt.plot(t2, y2, label="h = 0.1")
plt.plot(t3, y3, label="h = 0.01")
plt.plot(t4, y4, label="RK4 Default")
plt.plot(t5, y5, '--', label="RK4 Smooth", color="purple")
plt.xlabel("t"); plt.ylabel("y"); plt.legend()
plt.savefig("myEuler v RK4")

# -- Problem 2 -- #
def vanderPol(t, y):
    mu = 2
    x, xdot = y
    xdot = xdot
    xddot = xdot*mu*(1 - x**2) - x
    return xdot, xddot

init_cond = [1, 2]
myTimes = np.arange(0, 10, 1e-4)
sol = solve_ivp(vanderPol, (0, 10), init_cond, t_eval=myTimes)
x_values = sol.y[0]
xdot_values = sol.y[1]

plt.plot(myTimes, x_values, label="x(t)")
plt.plot(myTimes, xdot_values, label="x'(t)")
plt.xlabel("t"); plt.ylabel("x"); plt.legend()
plt.savefig("Van Der Pol ")

# -- Problem 3 -- #
def bounce(t, y, params):
    x, xdot, y, ydot = y
    m, f, b, k, g = params
    if y<= 0:   # Ball is in contact with the ground
        xdot = xdot
        xddot = (-f*xdot*math.sqrt(xdot**2 + ydot**2)) / m
        ydot = ydot
        yddot = (-f*ydot*math.sqrt(xdot**2 + ydot**2) - m*g - b*ydot - k*y) / m
    else:   # Ball is NOT in contact with the ground
        xdot = xdot
        xddot = (-f*xdot*math.sqrt(xdot**2 + ydot**2)) / m
        ydot = ydot
        yddot = (-f*ydot*math.sqrt(xdot**2 + ydot**2) - m*g) / m

    return xdot, xddot, ydot, yddot

params1 = (0.0027, 0.0007, 0.01, 200, 9.81)   # m, f, b, k, g
myTime = np.arange(0, 10, 1e-4)
sol_bounce = solve_ivp(bounce, (0, 10), [0, 1, 0.25, 0], t_eval=myTime, args=(params1,))
x_trajectory = sol_bounce.y[0]
y_trajectory = sol_bounce.y[2]
plt.plot(x_trajectory, y_trajectory)
plt.xlim((0, 1))
plt.title("Trajectory of Ping Pong Ball\nInitial Conditions: $x=0, \dot{x}=1, y=0.25, \dot{y}=0$")
plt.xlabel("Distance (m)"); plt.ylabel("Height (m)")
plt.savefig("Ping Pong Ball Bounce")