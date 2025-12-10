# Final Project
# ME EN 275
# Author: McKay Hoffmann
# Date: 12/9/2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

# Import Data
data = pd.read_csv("Spacex.csv")
data[["Date", "PAYLOAD_MASS__KG_"]] # Visualizing data

# Filter Data
data2019 = data.loc[0:11, "PAYLOAD_MASS__KG_"]
data2020 = data.loc[12:35, "PAYLOAD_MASS__KG_"]

# Summary Statistics
xbar2019 = np.mean(data2019)
xbar2020 = np.mean(data2020)
median2019 = np.median(data2019)
median2020 = np.median(data2020)
stdd2019 = np.std(data2019, ddof=1) # ddof = 1 because this is a sample, not a population
stdd2020 = np.std(data2020, ddof=1)
var2019 = stdd2019**2
var2020 = stdd2020**2

print(f"xbar2019 = {xbar2019}")
print(f"xbar2020 = {xbar2020}")
print(f"median2019 = {median2019}")
print(f"median2020 = {median2020}")
print(f"stdd2019 = {stdd2019}")
print(f"stdd2020 = {stdd2020}")
print(f"var2019 = {var2019}")
print(f"var2020 = {var2020}")

# Box and Whisker plots
plt.boxplot([data2019, data2020], labels=["2019", "2020"])
plt.ylabel("Payload Mass (kg)"); plt.legend()
plt.savefig("SpaceX Payload Mass Boxplots")

# Differencing 2-sample statistics
xbar = xbar2020 - xbar2019
n2019 = len(data2019)
n2020 = len(data2020)
stderr = math.sqrt((stdd2019**2)/n2019 + (stdd2020**2)/n2020)
dof = n2019 - 1 + n2020 - 1

# Confidence Interval
interval = stats.t.interval(0.95, dof, xbar, stderr)    # (np.float64(-452.16), np.float64(7196.77))
# Double checking accuracy of stats.t.interval
z = stats.t.ppf(0.025, df=dof)
upper_b = xbar + (-z * stderr)
lower_b = xbar + (z * stderr)

# Hypothesis Test (1-sided)
P_value = 1 - stats.t.cdf(xbar, dof, 0 ,stderr) # 0.04099214919181704