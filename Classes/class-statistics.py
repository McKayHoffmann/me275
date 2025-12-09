# Class: Statistics

# Histogram example
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
x = np.random.normal(0, 1, 100) # Mean, SD, samples
x1 = np.random.normal(0, 1, 100) # Mean, SD, samples
plt.hist(x, bins=10)    # Rule of thumb: # of bins you should aim for is the sqrt of the number of points you have
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.hist(x, bins=10, density=True)# Density=True changes the y axis from frequency to probability
plt.xlabel("Value")
plt.ylabel("Probability density")


plt.hist(x, [-3,-2,-1,0,1,2,3], density=True, alpha=0.5)# Density=True changes the y axis from frequency to probability
plt.hist(x1, [-3,-2,-1,0,1,2,3], density=True, alpha=0.5, color="red")
plt.xlabel("Value")
plt.ylabel("Probability density")

# Correlation number example
x = np.linspace(0, 1, 10)
y = np.array([1,2,4,5,8,12,18,20,40,50])
plt.scatter(x, y)
r = stats.pearsonr(x, y)[0]

