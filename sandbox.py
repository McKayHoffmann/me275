import scipy.stats as stats
import numpy as np
import math
xbar = 101
stdd = 5
n = 30
stderr = stdd / math.sqrt(n)
mu = 100

p_value = 1 - stats.norm.cdf(xbar, mu, stderr)

#
import matplotlib.pyplot as plt
import numpy as np
hours = np.arange(0, 9, 0.1)
n = len(hours)
tset1 = np.zeros_like(hours) + 2748.24
win = np.zeros_like(hours) + 916.08
cost = lambda x: (33.80/9) * x
cost1 = np.array([cost(xi) for xi in hours])
tset2 = np.zeros_like(hours) 

fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

ax1.plot(date, temperature, color=COLOR_TEMPERATURE, lw=3)
ax2.plot(date, price, color=COLOR_PRICE, lw=4)

ax1.set_xlabel("Date")
ax1.set_ylabel("Temperature (Celsius °)", color=COLOR_TEMPERATURE, fontsize=14)
ax1.tick_params(axis="y", labelcolor=COLOR_TEMPERATURE)

ax2.set_ylabel("Price ($)", color=COLOR_PRICE, fontsize=14)
ax2.tick_params(axis="y", labelcolor=COLOR_PRICE)

fig.suptitle("Temperature down, price up", fontsize=20)
fig.autofmt_xdate()

plt.show()