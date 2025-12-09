# Two-sided confidence interval example
import math
n = 50
xbar = 12
s_x = 1
stderr = s_x / math.sqrt(n)

import scipy.stats as stats
stats.norm.interval(.80, xbar, stderr)

# One-sided confidence interval
n = 30
xbar = 10
s_x = 1.3
stderr = s_x / math.sqrt(n)

z = stats.norm.ppf(.10, xbar, stderr)
