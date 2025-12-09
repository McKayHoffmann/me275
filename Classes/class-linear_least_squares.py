# Linear Least Squares
# ME 275
# Author: McKay Hoffmann
# Date: 9-30-25

# Most common use of least-squares method is to fit data
# We want to minimize (e_1)^2 + (e_2)^2 + (e_3)^2 + (e_4)^2


# Error = (y1-ax1-b)^2 + (y2-ax2-b)^2 + (y3-ax3-b)^2 + (y4-ax4-b)^2
#   The above is a 2D function, so we can find a minimum easily
# We want to minimize Error
# In other, easier words, instead of finding A*x = b, we want to minimize ||A*x - b||^2
# How to solve
import numpy as np
A = np.array([])
b = np.array([])
np.linalg.lstsq(A, b)

# Data fitting example - in class
import numpy as np
import matplotlib.pyplot as plt
x = np.array([-1, 0, 1.5, 3])
y = np.array([3, 1, 3, 8])
A = np.column_stack((x**2, x,  np.ones_like(x)))
b = y

coefficients = np.linalg.lstsq(A, b)[0]
x4plotting = np.linspace(-1, 3, 100)
y4plotting = np.array([coefficients[0]*xi**2 + coefficients[1]*xi + coefficients[2] for xi in x4plotting])

plt.scatter(x, y)
plt.plot(x4plotting, y4plotting)