# Class: Central Limit Theorem

# Copper wire example
import math
import scipy.stats as stats
mu = 0.66
stdd = 0.5244

muxbar = 0.66
stde = stdd / math.sqrt(100)
stats.norm.cdf(0.5, muxbar, stde)

# Confidence Intervals
# xbar +- z * s/sqrt(n)

# Heigh example
xbar = 67
se = 6
z = stats.norm.ppf(0.95)
stats.norm.cdf(1.64485)
stats.norm.cdf(0)