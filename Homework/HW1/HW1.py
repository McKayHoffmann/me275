# Homework 1
# ME EN 275
# Author: McKay Hoffmann
# Date: 9-8-2025

# import packages
import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import empty

# -- Problem 2 -- #
x1 = np.linspace(0, 11, 250)
x2 = np.linspace(11, 20, 250)
x3 = np.linspace(20, 32, 250)
x4 = np.linspace(32, 47, 250)

y1 = 288.15 - 6.5*x1
y2 = np.zeros(250) + 216.65
y3 = 216.65 + (x3 - 20)
y4 = 228.65 + 2.8 * (x4 - 32)

# x = np.vstack([x1, x2, x3, x4])
# y = np.vstack([y1, y2, y3, y4])

# fig = plt.figure()
plt.plot(y1, x1, color="red")
plt.plot(y2, x2, color="red")
plt.plot(y3, x3, color="red")
plt.plot(y4, x4, color="red")
plt.ylabel("Altitude (km)")
plt.xlabel("Temperature (K)")
plt.title("Altitude v. Temperature")
plt.savefig("AltitudeVsTemperature.png")
# plt.show()

# -- Problem 3 -- #
def polar2cart(r, theta):
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    return x, y

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x))
    return r, theta

r = 10
theta = 33 # degrees

x, y = polar2cart(r, theta)
print(x, y)
r_test, theta_test = cart2polar(x, y)
print(r_test, theta_test)

# Requirements for how results are to be returned were not given in the homework
# Too bad Python doesn't have multiple dispatch
def polar2cart_np(r, theta):
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    return x, y

def cart2polar_np(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x))
    return r, theta

x_vec = np.array([1, 2, 3])
y_vec = np.array([1, 2, 3])

r_vec, theta_vec = cart2polar(x_vec, y_vec)
print(r_vec, theta_vec)
x_vec_test, y_vec_test = polar2cart_np(r_vec, theta_vec)
print(x_vec_test, y_vec_test)

# -- Problem 4 -- #
def avg(x):
    return sum(x) / len(x)

x = np.array([1, 2, 3])
print(avg(x))

def mavg(x, c=1):
    return c * avg(x)

print(mavg(x))
print(mavg(x, 5))

def wavg(x, w):
    return sum(x * w)/ sum(w)

x = np.array([4, 3, 2])
w = np.array([3, 2, 1])
print(wavg(x, w))

def tavg(filename):
    import os
    os.chdir(r"C:\Users\mckay\OneDrive\Documents\Code\ME EN 275\Homework")
    data = np.loadtxt(filename, delimiter=',')
    for i in range(data.shape[1]):
        print(f"Average of column {i+1}: {avg(data[:, i])}")

tavg("HW1_tavg_test.csv")