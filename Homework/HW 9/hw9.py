# HW 9
# ME 275
# Author: McKay Hoffmann
# Date: 11/20/25
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

# -- Problem 1 -- #
# Visualizing data
value = np.array([1,2,3,4,5])
probability = np.array([0.4, 0.25, 0.2, 0.1, 0.05])
plt.bar(value, probability)
sum(probability)    # 1.0

mu = sum(value*probability) # 2.15
variance = sum(probability * (value - mu)**2)   # 1.4275

# -- Problem 2 -- #
import scipy.integrate as intg

def prob_mass_fun(x):
    if 80 < x and x < 120:
        return (x - 80) / 800
    else:
        return 0
    
def prob_mass_4_mu(x):
    if 80 < x and x < 120:
        return x * ((x - 80) / 800)
    else:
        return 0
    
def prob_mass_4_std(x, mu2=106.66666668320516):
    if 80 < x and x < 120:
        return ((x - mu2)**2 * ((x - 80) / 800))
    else:
        return 0

# Visualizing data to double check intuition
x_feeder = np.arange(79, 121, 1e-4)
prob_data = np.array([prob_mass_fun(xi) for xi in x_feeder])
plt.plot(x_feeder, prob_data)
# Integral should equal ~1
total_prob = intg.quad(prob_mass_fun, 80, 120)  # 0.99999

# Proportion of resistors that have resistances less that 90 Ohms
prop_a = intg.quad(prob_mass_fun, 0, 90)    # 0.0625
# Mean resistance
mu2 = intg.quad(prob_mass_4_mu, 0, 130) # 106.66666668320516
# Standard deviation
stdd = math.sqrt(intg.quad(prob_mass_4_std, 0, 130)[0]) # 9.428090416239776

# -- Problem 3 -- #
# a
n = 123
xbar = 136.9
stdd = 22.6
stderr = stdd / math.sqrt(n)
interval = stats.norm.interval(.95, 136.9, stderr)  # 132.906, 140.894
# b
z = (139.9 - 136.9) * (math.sqrt(123) / 22.6)
confidence = stats.norm.cdf(z) - stats.norm.cdf(-z) # 0.8590318028563106
# Double checking
print(stats.norm.interval(confidence, xbar, stderr))
# c
necessary_z_score = stats.norm.ppf(0.95+(0.05/2))
necessary_sample_size = (stdd * necessary_z_score / 3)**2   # 218.00705636197017
# Double checking
print(stats.norm.interval(0.95, xbar, stdd/math.sqrt(necessary_sample_size)))
# d
lower_bound = stats.norm.ppf(0.02, xbar, stderr)    # 132.71493

# -- Problem 4 -- #
n = 5
octane_data = np.array([87.0,86.0,86.5,88.0,85.3])
xbar = np.mean(octane_data)
stdd = np.std(octane_data, ddof=1)
stderr = stdd / math.sqrt(n)
interval = stats.t.interval(0.99, 4, xbar, stderr)  # 84.457, 88.663

# -- Problem 5 -- #
import matplotlib.pyplot as plt
datam = np.genfromtxt('ANSUR II MALE Public.csv', delimiter=',', names=True)
knee_m = datam['kneeheightmidpatella']
dataf = np.genfromtxt('ANSUR II FEMALE Public.csv', delimiter=',', names=True, encoding='UTF-8')
knee_f = dataf['kneeheightmidpatella']
kneedata = np.concatenate((knee_m, knee_f), axis=None)
bins = round(math.sqrt(len(kneedata)))
# Regular histogram
plt.hist(kneedata, bins=bins)
plt.xlabel("Knee Height"); plt.ylabel("Frequency"); plt.legend()
plt.title("Histogram for Knee Height")
plt.savefig("Histogram for Knee Height.png")
# Probability Distribution
plt.hist(kneedata, bins=bins, density=True)
plt.xlabel("Knee Height"); plt.ylabel("Frequency"); plt.legend()
plt.title("Probability Distribution for Knee Height")
plt.savefig("Probability Distribution for Knee Height.png")
# Cumulative Distribution
plt.hist(kneedata, bins=bins, cumulative=True, density=True)
plt.xlabel("Knee Height"); plt.ylabel("Frequency"); plt.legend()
plt.title("Cumulative Distribution for Knee Height")
plt.savefig("Cumulative Distribution for Knee Height.png")

# -- Problem 6 -- #
x = np.linspace(min(kneedata), max(kneedata), int(math.sqrt(len(kneedata))))
knee_mean = np.mean(kneedata)
knee_stdd = np.std(kneedata, ddof=1)
pdf = stats.norm.pdf(x, knee_mean, knee_stdd)
plt.hist(kneedata, bins=bins, density=True)
plt.plot(x, pdf)
plt.xlabel("Knee Height"); plt.ylabel("Frequency"); plt.legend()
plt.title("Probability Distribution for Knee Height")
plt.savefig("Probability Distribution for Knee Height with fit.png")