# Gear coating example
import scipy.stats as stats
import math
import numpy as np

p = stats.norm.cdf(74, 80, (18/math.sqrt(60)))

# A formal procedure
# 1. Define null and alternative hypotheses, and choose alpha.
# 2. We assume the null is true
# 3. Collect data. Compute P-value
# 4. Compare P-value to alpha. Either reject or fail to reject the null hypothesis

# Transmission example
data = np.array([39.030, 38.997, 39.012, 39.008, 39.019, 39.002])
xbar = np.mean(data)
stdd = np.std(data)

p = 2*(1 - stats.t.cdf(xbar, 5, 39, (stdd/math.sqrt(6))))

# PART 2
mu_diff = 44.1 - 32.3
theta_d = math.sqrt(10.09**2 / 10 + 8.56**2 / 10)
p = 1 - stats.t.cdf(mu_diff, 18, 0, theta_d)
lower_bound = stats.t.ppf(0.05, 18, mu_diff, theta_d)

data = np.array([0,0,1,3,4,0,0,5,5,1,1,1,13])
print(np.std(data, ddof=1))
print(np.var(data, ddof=1))
