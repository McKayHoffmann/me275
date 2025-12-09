# Bicycle Example
import math
zbar = 2250 - 2000

# New Car Tire Example
import numpy as np
import scipy.stats as stats
data = np.array([0.16,0.38,0.17,0.31,0.19,0.35,0.43,-0.21,0.34,0.20])
mean = np.mean(data)
std = np.std(data, ddof=1)
stderr = std / math.sqrt(10)

p_value = 1 - stats.t.cdf(mean, 9, 0, stderr)
print(f"P-value = {p_value}")

lower_bound = stats.t.ppf(0.05, 9, mean, stderr)
print(f"Lower Bound = {lower_bound}")