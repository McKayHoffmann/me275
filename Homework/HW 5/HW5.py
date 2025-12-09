# HW 5
# ME 275
# Author: McKay Hoffmann
# Date: 10/9/2025

import numpy as np
import math
import matplotlib.pyplot as plt

# -- Problem 1 -- #
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
b = np.array([8, 3, 5])
C = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
# a
A + C.T
# b
2 * A
# c
A @ b
# d
b.T @ A.T
# e
A @ A.T

# -- Problem 2 -- #
cosd = lambda degrees: math.cos(math.radians(degrees))
sind = lambda degrees: math.sin(math.radians(degrees))

def getVector(head, tail):
    return head - tail

def getAngle(vec1, vec2):
    return math.degrees(math.acos((np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))

def getAngles(JointA, JointB, JointC, vecAB=np.array([1, 0])):
    vecAC = getVector(JointC, JointA)
    vecBC = getVector(JointC, JointB)
    alpha = getAngle(vecAC, vecAB)
    beta = getAngle(-vecAB, vecBC)
    gamma = 180 - alpha - beta
    a = getAngle(-vecBC, np.array([0, -1]))
    b = gamma + a
    c = 180 - beta
    return alpha, beta, gamma, a, b, c

def getA(a, b, c, alpha):
    A = np.array([
    [0, 0, 0, -cosd(b), -cosd(a), 0],
    [0, 0, 0, -sind(b), -sind(a), 0],
    [0, 0, 1, 0, sind(c), 0],
    [0, 0, 0, 0, cosd(c), -1],
    [0, 1, 0, sind(alpha), 0, 0],
    [1, 0, 0, cosd(alpha), 0, 1]
    ])
    return A

JointA = np.array([0, 0])
JointB = np.array([1, 0])
JointC = np.array([2, 1])
alpha, beta, gamma, a, b, c = getAngles(JointA, JointB, JointC)

A = getA(a, b, c, alpha)
vecb = np.array([1000, 0, 0, 0, 0, 0])
x = np.linalg.solve(A, vecb)   # Returns [F_Ax, F_Ay, F_B, T_AC, T_BC, T_AB]
x[4]
# T_BC = -2828.427

## Get plot for tension in AC as xc is varied
xc = np.linspace(1, 3, 500)
tensionLog = np.array([])
vecb = np.array([1000, 0, 0, 0, 0, 0])

for xci in xc:
    JointC = np.array([xci, 1])
    alpha, beta, gamma, a, b, c = getAngles(JointA, JointB, JointC)
    A = getA(a, b, c, alpha)
    x = np.linalg.solve(A, vecb)
    tensionLog = np.append(tensionLog, x[3])

# yticks = np.arange(0, 6000+500, 500)
plt.grid(True)
plt.plot(xc, tensionLog, label="Tension in member AC")
# plt.yticks(yticks)
plt.xlabel("$x_c$", labelpad=10)    # Python interprets anything inside of $$'s as LaTeX math 
plt.ylabel("Tension in member AC ($lbs$)", labelpad=10)     # labelpad moves the label
plt.savefig("Tension in AC")

# How to find the location of an element you already know exists
# arr = np.array([10, 20, 30, 40])
# index = np.where(arr == 30)[0][0]  # Returns 2
# index

# -- Problem 3 -- #
k1 = 100    # N/m
k2 = 150
k3 = 100
g = 9.81
A = np.array([
    [-k3, -k3, k3],
    [-k2, k2, 0],
    [k1, 0, 0]
])
b = np.array([2*g, 5*g, 10*g])
x = np.linalg.solve(A, b)
print(x)

# -- Problem 4 -- #
x = np.array([-2.0000, -1.7895, -1.5789, -1.3684, -1.1579, -0.9474, -0.7368, -0.5263, -0.3158, -0.1053, 0.1053, 0.3158, 0.5263, 0.7368, 0.9474, 1.1579, 1.3684, 1.5789, 1.7895, 2.0000])
y = np.array([7.7859, 5.9142, 5.3145, 5.4135, 1.9367, 2.1692, 0.9295, 1.8957, -0.4215, 0.8553, 1.7963, 3.0314, 4.4279, 4.1884, 4.0957, 6.5956, 8.2930, 13.9876, 13.5700, 17.7481])
A = np.column_stack((x**2, x, np.ones_like(x)))
b = y
results = np.linalg.lstsq(A, b)
results[0][0]
a, b, c = results[0][0], results[0][1], results[0][2]
xx = np.arange(np.min(x), np.max(x), 0.01)
yy = a*xx**2 + b*xx + c
plt.plot(x, y, "o")
plt.plot(xx, yy)
plt.savefig("Least Squares")
print(a)

# -- Problem 3 Correction -- #
import numpy as np
k1 = 100
k2 = 150
k3 = 100
g = 9.81
A = np.array([
    [-k3, -k3, k3],
    [-k2, k2, 0],
    [k1, 0, 0]
])
b = np.array([2*g, 5*g, 10*g])
x = np.linalg.solve(A, b)
x


k1 = 100
k2 = 150
k3 = 100
g = 9.81
A = np.array([
    [-k3, -k3, k3],
    [(-k2 + k3), (k2 + k3), -k2],
    [(k1 + k2), -k2, 0]
])
b = np.array([2*g, 3*g, 5*g])
x = np.linalg.solve(A, b)
x