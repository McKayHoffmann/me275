# HW 8
# ME 275
# Author: McKay Hoffmann
# Date: 11/6/25
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

# -- Problem 1 -- #
datam = np.genfromtxt('ANSUR II MALE Public.csv', delimiter=',', names=True)
knee_m = datam['kneeheightmidpatella']
dataf = np.genfromtxt('ANSUR II FEMALE Public.csv', delimiter=',', names=True, encoding='UTF-8')
knee_f = dataf['kneeheightmidpatella']
kneedata = np.concatenate((knee_m, knee_f), axis=None)

plt.hist(kneedata, bins=78)
plt.xlabel("Length $(mm)$"); plt.ylabel("Frequency"); plt.title("Knee Height Midpatella")
plt.savefig("Knee Height Histogram")

filteredkneedata = kneedata[(kneedata >= 425) & (kneedata <= 550)]
# Checking that it filtered properly
plt.hist(filteredkneedata, bins=76)

plt.boxplot(filteredkneedata)
perc0 = np.percentile(filteredkneedata, 0)      # 425.0 mm
perc25 = np.percentile(filteredkneedata, 25)    # 456.0 mm
perc50 = np.percentile(filteredkneedata, 50)    # 478.0 mm
perc75 = np.percentile(filteredkneedata, 75)    # 498.0 mm
perc100 = np.percentile(filteredkneedata, 100)  # 550.0 mm
# Size 1: 425-456 mm
# Size 2: 456-478 mm
# Size 3: 478-498 mm
# Size 4: 498-550 mm

# Extracting Stature Data
stature_m = datam['stature']
stature_f = dataf['stature']
staturedata = np.concatenate((stature_m, stature_f), axis=None)
# Correlation Coefficient
r = stats.pearsonr(staturedata, kneedata)[0]    # 0.88936
# Finding Least Square Fit
A = np.column_stack((staturedata, np.ones_like(staturedata)))
a, b = np.linalg.lstsq(A, kneedata)[0]
# Plotting
lst_sqr = (a * staturedata) + b
plt.scatter(staturedata, kneedata, s=5)
plt.plot(staturedata, lst_sqr, color="red")
plt.xlabel("Stature $(mm)$"); plt.ylabel("Knee Height Mid-Pattela $(mm)$")
plt.savefig("Mid-Patella v Stature scatterplot.png")

# -- Problem 2, 3, and 4 -- #
# Work done on scratch paper