# Linear Algebra
# ME 275
# Author: McKay Hoffmann
# Date: 9-25-25

## VECTORS
import numpy as np
x = np.array([1, 2, 3]) 
x.shape # This is neither a row or column vector
y = x.reshape((1, 3))
y.shape # Row vector
z = x.reshape((3, 1))
z.shape # Column vector
np.transpose(y)
# or
y.T

## Dot product
# Scalars are greek symbols in this class
# lower case is a vector
# capital letters is a matrix
y = x
alpha = np.dot(x, y)
alpha

## Norm
# alpha = ||x|| = sqrt(x1^2 + x2^2 + ... + xn^2)
alpha = np.linalg.norm(x)
alpha

## Matrix vector multiplication
# How to create a matrix 
A = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
A
# A * x = b
y = np.dot(A, x)
y
# Vector matrix multiplication
# y.T = x.T * A
y2 = np.dot(x, A)
y2

## Matrix Matrix Multiplication
B = A
np.dot(A, B)
C = A @ B
C

## Linear System of Equations
# A * x = b
# A and b are known
# x is the unknown
# x = A^-1 * b
# DO NOT USE THE ABOVE!!!!
x = np.linalg.solve(A, b)

# Triangular Systems
# You can always take a square matrix and break it up into a lower triangular and an upper triangular

# Example problem
A = np.array([[-0.27, 0.32, -0.16], [-0.8, -0.95, -0.97], [-0.53, 0, 0.16]])
b = np.array([0, -5, 0])
x = np.linalg.solve(A, b)
x
A @ z


## Linear Algebra Continuted ##
# Heat transfer example
# Every node is the average of its 4 neighbors
import numpy as np
A = np.array([[-4,1,1,0], [1,-4,0,1], [1,0,-4,1], [0,1,1,-4]])
b = np.array([-30,-60,-40,-70])
x = np.linalg.solve(A, b)

# Statics example
import math
sind = lambda theta: math.sin(math.radians(theta))
cosd = lambda theta: math.cos(math.radians(theta))
A = np.array([[-sind(60),0,sind(30),0,0,0], 
              [-cosd(60),0,-cosd(30),0,0,0],
              [cosd(30),1,0,1,0,0], 
              [sind(30),0,0,0,1,0],
              [0,-1,-cosd(60),0,0,0], 
              [0,0,sind(60),0,0,1]])
b = np.array([0,1000,0,0,0,0])
x = np.linalg.solve(A, b)