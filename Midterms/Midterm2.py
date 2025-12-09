# Midterm 2
# ME 275
# Author: McKay Hoffmann
# Date: 11/11/2025
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
import scipy.stats as stats

# -- Problem 1 -- #
def odefun(t, y):
    return (y*t**2) - 1.1*y

def myEuler(fun, init_cond, h, interval):
    t_range = np.arange(interval[0], interval[1]+h, h)
    y_log = np.array([init_cond])
    for ti in t_range:
        yi1 = y_log[-1] + (fun(ti, y_log[-1]))*h
        y_log = np.append(y_log, yi1)
    return t_range, y_log[:-1]

t_range, y_log = myEuler(odefun, 1, 0.5, (0,2))
t_range[-1] # 2.0
y_log[-1]   #  0.3871546874999998

# -- Problem 2 -- #
def func(t, y):
    dydt = (y * t**2) - 1.1*y
    return dydt

my_time = np.arange(0, 2, 1e-6)
sol = solve_ivp(func, (0,2), [1], t_eval=my_time)
sol.y[0][-1]    # 1.5948534147099174
sol.t[-1]       # 1.9999989999999999
# Convergance study
my_time = np.arange(0, 2, 1e-8)
sol = solve_ivp(func, (0,2), [1], t_eval=my_time)
sol.y[0][-1]    # 1.5948579935377816
sol.t[-1]       # 1.99999999
len(sol.y[0])

# -- Problem 3 -- #
def odesystem(t, states):
    y, z = states
    dydt = -2*y + 4*math.exp(-t)
    dzdt = -(y*z**2)/3
    return dydt, dzdt

my_time3 = np.arange(0, 4, 1e-6)
sol3 = solve_ivp(odesystem, (0, 4), [2, 4], t_eval=my_time3)
ydata = sol3.y[0]
zdata = sol3.y[1]

plt.plot(sol3.t, ydata, label="State y")
plt.plot(sol3.t, zdata, label="State z")
plt.legend(); plt.xlabel("Time"); plt.ylabel("State value")
plt.savefig("Midterm 2 Problem 3")

# -- Problem 4 -- #
def wall_temp(x, states):
    t, tprime = states
    dtdx = tprime
    if 0.0 <= x and x < 0.05:
        k = 0.1
        ddtdx = (-tprime - t) / k
    elif 0.05 <= x and x <= 0.2:
        k = 0.2
        ddtdx = (-tprime - t) / k
    return dtdx, ddtdx
    
init_conditions4 = (0, 100)

my_time4 = np.arange(0, 0.2, 1e-7)
sol4 = solve_ivp(wall_temp, (0, 0.2), init_conditions4, t_eval=my_time4)
Tdata = sol4.y[0]
plt.plot(sol4.t, Tdata, label="Temperate")
plt.legend(); plt.xlabel("Distance (x)"); plt.ylabel("Temperature")
sol4.y[0][-1]   # 9.976309653199774
sol4.t[-1]  # 0.19999999999999998

# Convergance study
my_time4_1 = np.arange(0, 0.2, 1e-8)
sol4_1 = solve_ivp(wall_temp, (0, 0.2), init_conditions4, t_eval=my_time4_1)
Tdata_1 = sol4_1.y[0]
sol4_1.y[0][-1]   # 9.97630940924706
sol4_1.t[-1]  # 0.19999999000000002

# -- Problems 5-10 -- #
data = np.array([0, 0, 1, 3, 4, 0, 0, 5, 5, 1, 1, 1, 13])
np.mean(data)   # 2.6153846153846154
np.median(data) # 1.0
np.percentile(data, 80) # 4.600000000000001
np.std(data)    # 3.4980975387390996
np.var(data)    # 12.236686390532546
plt.hist(data, bins=10)
plt.savefig("Midterm 2 Problem 10")

# -- Problems 11-13 -- #
temperatures = np.arange(40, 95+5, 5) # Degrees Celcius
lifetimes = np.array([851, 635, 764, 708, 469, 661, 586, 371, 337, 245, 129, 158])    # hours
plt.scatter(temperatures, lifetimes)
plt.xlabel("Temperatures (C)"); plt.ylabel("Lifetime (hours)")
r = stats.pearsonr(temperatures, lifetimes)[0]  # -0.9346328250899915
A = np.column_stack((temperatures, np.ones_like(temperatures)))
coefficients = np.linalg.lstsq(A, lifetimes)[0]
a, b, = coefficients[0], coefficients[1]    # a = -12.611188811188807, b = 1344.0885780885776
# Double checking
lstsqdata = np.array([(a*ti + b) for ti in temperatures])
plt.scatter(temperatures, lifetimes)
plt.plot(temperatures, lstsqdata)
plt.xlabel("Temperatures (C)"); plt.ylabel("Lifetime (hours)")

# -- Problems 14-22 -- #
# Done on scratch paper

# -- Problem 21 -- #
# Need the weighted average. If there were 10,000 men and 30 women, the combined
# average would be different that if there were 10 men and 30 women, assuming the
# same averages.
averages = np.array([178, 164])
weights = np.array([20, 30])
np.average(averages, weights=weights)   # 169.6 cm
(178 + 164) / 2 # 171.0 cm  -- DIFFERENT!

# -- Problems 23-24 -- #
import scipy.integrate as integrate
t = [1, 2, 3.25, 4.5, 6, 7, 8, 8.5, 9, 10]
v = [5, 6, 5.5, 7, 8.5, 8, 6, 7, 7, 5]
plt.plot(t, v)
distance_travelled = integrate.trapezoid(v, t)  # 60.125
t[-1] - t[0]    # b - a = 9
average = distance_travelled / 9    # 6.680555555555555
# Confirming average
average_data = np.zeros_like(t) + average
distance_travelled_w_average = integrate.trapezoid(average_data, t) # 60.125
# Confirming that it is different
np.mean(v)  # 6.5

# -- Problem 28 -- #
data1 = np.array([0, 0, 1, 3, 4, 0, 0, 5, 5, 1, 1, 1, 13])
np.mean(data1)   # 2.6153846153846154
np.median(data1) # 1.0
np.std(data1)    # 3.4980975387390996
np.var(data1)    # 12.236686390532546

data2 = np.array([0, 0, 1, 3, 4, 0, 0, 5, 5, 1, 1, 1, 25])
np.mean(data2)   # 3.5384615384615383
np.median(data2) # 1.0
np.std(data2)    # 6.452374454003313
np.var(data2)    # 41.633136094674555

temperatures1 = np.arange(40, 95+5, 5) # Degrees Celcius
lifetimes1 = np.array([851, 635, 764, 708, 469, 661, 586, 371, 337, 245, 129, 158])    # hours
r = stats.pearsonr(temperatures1, lifetimes1)[0]  # -0.9346328250899915

temperatures2 = np.arange(40, 95+5, 5) # Degrees Celcius
lifetimes2 = np.array([1500, 635, 764, 708, 469, 661, 586, 371, 337, 245, 129, 158])    # hours
r2 = stats.pearsonr(temperatures2, lifetimes2)[0]  # -0.858851453978366