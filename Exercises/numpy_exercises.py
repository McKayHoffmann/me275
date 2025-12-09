import numpy as np

x = np.linspace(0 ,3, 30)
y = 3 * np.ones(10)

A  = np.array([[4, 6, 6, 9], [7, 4, 10, 5], [1, 6, 4, 10]])
dimensions_of_A = A.shape
print(A[1, 2])

# Store the third column of the above matrix to a new variable
third_column = A[:, 2]

# Print the last row
# Julia: A[end, :]
# Python: A[-1, :]
print(A[-1, :])

# Multiply the matrix by 2
print(A * 2)

# Extract all entries from A that are greater than 5 into a vector
A[A > 5]

# Define two arrays: a = [1, 2, 3] and b = [2, 2, 2]. See what happens
# when you multiply them together
a = np.array([1, 2, 3])
b = np.array([2, 2, 2])
x = a * b

# print the maximum entry in x
print(max(x))

# Stack the two vectors on top of eachother to create a matrix of size
# 2x3 (try vstack or concatenate)
C = np.vstack([a, b])

# Transpose C
C_transposed = C.T
print(C_transposed)
print(C)

# Reverse the order of a
print(np.flip(a))
print(a[::-1])

# (Challenge): calculate the mean squared error between a and b (subtract them,
# square the result, then compute average of result)
d = np.mean((a - b) ** 2)
print(d)