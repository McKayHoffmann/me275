# Midterm 1
# ME 275
# Author: McKay Hoffmann
# Date: 10/14/2025

# Problem 1
import numpy as np
import math
from scipy.optimize import root_scalar
residual = lambda x: x - 0.4*np.sin(x) - (math.pi / 2)
x = root_scalar(residual, bracket=(1,2), method='bisect').root
x1 = root_scalar(residual, x0=1, maxiter=2, method='Newton').root
x
x1
# Relative error: |(x - x\star) / x\star|
relative_error = abs((x1-x) / (x))*100
print(relative_error)

# Problem 4
T = np.array([10, -2, -15, -30])    # Celcius
t = np.array([0, 1.2, 4.8, 7.9])
slope = (-15 - 10) / 4.8
print(slope)    # Simple rise over run problem?

# Problem 5 and 6
f = lambda x: (math.sin(x) * x**2) + math.log(x)
def finite_differencing(function, x0, h=1e-6):
    return (function(x0 + h) - function(x0)) / (h)

fprime = finite_differencing(f, 2)
print(fprime)

def central_differencing(function, x0, h=1e-4):
    return (function(x0 + h) - function(x0-h)) / (2*h)

fprimecentral = central_differencing(f, 2)
absolute_error = abs(fprime - fprimecentral)
print(absolute_error)

# Problem 7 and 8
from scipy import integrate
g = lambda x: 5 * np.exp(-0.5 * x) * math.cos(2 * x)
x = np.linspace(-2, 1, 50)  # 50 points
len(x)
gs = np.array([g(xi) for xi in x])
len(gs)
integral = integrate.trapezoid(gs, x)
print(integral)
trueintegral = integrate.quad(g, -2, 1)[0]
absolute_error = abs(trueintegral - integral)
print(absolute_error)

# Problem 9
import matplotlib.pyplot as plt
xvec = np.array([0.0, 0.06411414, 0.12822827, 0.19234241, 0.25645654, 0.32057068, 0.38468481, 0.44879895, 0.51291309, 0.57702722, 0.64114136, 0.70525549, 0.76936963, 0.83348377, 0.8975979, 0.96171204, 1.02582617, 1.08994031, 1.15405444, 1.21816858, 1.28228272, 1.34639685, 1.41051099, 1.47462512, 1.53873926, 1.60285339, 1.66696753, 1.73108167, 1.7951958, 1.85930994, 1.92342407, 1.98753821, 2.05165235, 2.11576648, 2.17988062, 2.24399475, 2.30810889, 2.37222302, 2.43633716, 2.5004513, 2.56456543, 2.62867957, 2.6927937, 2.75690784, 2.82102197, 2.88513611, 2.94925025, 3.01336438, 3.07747852, 3.14159265])
yvec = np.array([-1.33420675, -1.10031441, -1.70601521, -0.44700296, 0.30883201, 0.01610921, 0.81291503, 0.95636478, 1.03921739, 0.99634815, 1.26036554, 1.52465377, 0.24763214, 0.99063965, 0.29047526, 0.067633, -1.06786088, -0.77228815, -0.41577237, -1.97700207, -2.65033565, -2.68936338, -1.76184177, -0.69284902, -1.22291395, -0.75623781, -1.26228219, -0.25055401, -0.65165001, 0.28173528, 0.37392696, 0.78813861, 1.68701433, 1.92216713, 3.314706, 3.19845057, 3.66274494, 3.55980745, 4.51291657, 3.55387125, 4.1570019, 3.42706402, 2.77786829, 2.52253828, 1.76818801, 1.36396786, 0.51341986, -0.01626634, -0.79174608, -2.07371013])
column1 = np.array([math.sin(xi) for xi in xvec])
column2 = np.array([math.sin(2 * xi) for xi in xvec])
column3 = np.array([math.sin(3 * xi) for xi in xvec])
column4 = np.ones_like(column1)
A = np.column_stack((column1, column2, column3, column4))
b = yvec
coefficients = np.linalg.lstsq(A, b)[0]
print(coefficients)
b = coefficients[1]
print(b)

h = lambda x: coefficients[0]*math.sin(x) + coefficients[1]*math.sin(2*x) + coefficients[2]*math.sin(3*x) + coefficients[3]
fitx = np.linspace(min(xvec), max(xvec), 100)
fity = np.array([h(xi) for xi in fitx])
plt.plot(xvec, yvec, "o")
plt.plot(fitx, fity)    # Confirmed, line of best fit

# Problem 10
A = np.array([
    [0,3,0,6],
    [1,0,2,0],
    [0,5,5,0],
    [0,0,7,1]
])
b = np.array([18, -5, -25, -17])
x = np.linalg.solve(A, b)
print(x)
A @ x

# Problem 11
A = np.array([
    [3,-1,-2],
    [-1,3,-1],
    [-2, -1, 5]
])
b = np.array([15,0,0])
x = np.linalg.solve(A, b)
print(x)
i3 = x[2]
print(i3)

# Problem 12
y = lambda x: math.sin(x**2)
xs = np.linspace(0, 2*math.pi, 1000)
ys = np.array([y(xsi) for xsi in xs])
cum_int = integrate.cumulative_simpson(y=ys, x=xs)
cum_int = np.insert(cum_int, 0, 0)
plt.plot(xs, ys, label="Original function")
plt.plot(xs, cum_int, label="Cumulative Integral")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="lower left")
plt.savefig("Cumulative Integral.png")

# Problem 13
def next_guess(yi, x):
    return (yi + (x / yi)) / 2

def approx_sqrt(x, tol=10e-6):
    yi = x
    yi1 = next_guess(yi, x)
    while abs(yi1 - yi) > tol:
        yi = yi1
        yi1 = next_guess(yi, x)
    return yi1

approx_sqrt(186)    # Returns 13.638181696985857
# Using a calculator: 13.638181697
# More than 4 sig figs of accuracy

# Problem 15: It could go around in a loop forever using derivatives
# Problem 16: It will likely never reach 0, or it will be very computationally expensive. Use a tolerance.

# Problem 17
# First of all, matrix multiplation needs @ operator, not *
A = np.array([
    [1, 2, 3],
    [3, 4, 5],
    [5, 6, 7]
])
B = np.array([
    [7, 8, 9],
    [9, 10, 11],
    [11, 12, 13]
])

A * B   # Same
B * A   # Same
A @ B   # Different
B @ A   # Different

# Problem 19: Run a convergance study
# Problem 20: NEVER invert the matrix. Use np.linalg.solve, which breaks A up into two triangular matrices and then solves
# Problem 21: Could have a free variable, or it could simply not have a solution at all
# Problem 22: When you subtract f(x) from f(x+h), roundoff error happens.

# Problem 23
from scipy.optimize import root_scalar
y = lambda x: math.sin(x)   # sin(x) contains multiple roots
root = root_scalar(y, bracket=((-math.pi / 2), (5 * math.pi / 2)), method="bisect")  # Contains 3 roots
root    # Still works
