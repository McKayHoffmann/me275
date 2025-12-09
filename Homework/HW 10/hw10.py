# HW 10
# ME EN 275
# Author: McKay Hoffmann
# Date: 12/2/2025
import math
import scipy.stats as stats
import numpy as np

# -- Problem 1 -- #
mu = 15 # micrometers
n = 87
xbar = 15.2
stdd = 1.8
stderr = stdd / math.sqrt(n)

p_value = 2 * (1 - stats.norm.cdf(xbar, mu, stderr))

# -- Problem 2 -- #
mu = 16
data = np.array([15.87, 16.02, 15.78, 15.83, 15.69, 15.81, 16.04, 15.81, 15.92, 16.10])
xbar = np.mean(data)
n = len(data)
stdd = np.std(data, ddof=1)
stderr = stdd / math.sqrt(n)

z = stats.t.ppf(0.95, df=(n-1))
# z = stats.norm.ppf(0.95)
upper_b = xbar + (z * stderr)

p_value = stats.t.cdf(xbar, 9, mu, stderr)

# -- Problem 3 -- #
xbar1 = 169.9
stdd1 = 24.8
n1 = 110
xbar2 = 163.3
stdd2 = 25.8
n2 = 225

xbar = abs(xbar1 - xbar2)
stderr = math.sqrt((stdd1**2)/n1 + (stdd2**2)/n2)
interval = stats.norm.interval(0.95, xbar, stderr)

p_value = 2*(1 - stats.norm.cdf(xbar, 0, stderr))

# -- Problem 4 -- #
disk_data = np.array([269.0, 249.3, 255.2, 252.7, 247.0, 261.6])
oval_data = np.array([268.8, 260.0, 273.5, 253.9, 278.5, 289.4, 261.6, 280.2])
mean_disk = np.mean(disk_data)
mean_oval = np.mean(oval_data)
n_disk = len(disk_data)
n_oval = len(oval_data)
stdd_disk = np.std(disk_data, ddof=1)
stdd_oval = np.std(oval_data, ddof=1)

xbar = abs(mean_disk - mean_oval)
stderr = math.sqrt((stdd_disk**2)/n_disk + (stdd_oval**2)/n_oval)
dof = n_disk + n_oval - 2

p_value = 2*(1 - stats.t.cdf(xbar, dof, 0, stderr))

# -- Problem 5 -- #
n = 10
rest = np.array([15, 16, 21, 17, 18, 15, 19, 21, 18, 14])
exercise = np.array([30, 37, 39, 37, 40, 39, 34, 40, 38, 34])
diff = exercise - rest
xbar = np.mean(diff)
stdd = np.std(diff, ddof=1)
stderr = stdd / math.sqrt(n)

interval = stats.t.interval(0.95, 9, xbar, stderr)