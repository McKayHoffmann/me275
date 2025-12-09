import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 10)
y = np.sin(x)

plt.plot(x, y)
plt.show()
plt.plot(x, y, "-")
plt.show()


def function(x):
    return x * 2

'''
Class notes - NUMERICAL ERRORS

Trucation Error:

Roundoff Error:

Modeling Error:
- Our model is inaccurate, it doens't accurately describe reality

Coding Error:
- Your code has a bug in it

Absolute Error: \epsilon = |x - x\star|   x\star is the true value
Relative Error: \epsilon = |(x - x\star) / x\star|

'''
