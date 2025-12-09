# Final Project
# ME EN 275
# Author: McKay Hoffmann
# Date: 12/9/2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Data
data = pd.read_csv("Spacex.csv")
data[["Date", "PAYLOAD_MASS__KG_"]]

# Filter Data
data2019 = data.loc[0:11, "PAYLOAD_MASS__KG_"]
data2020 = data.loc[12:35, "PAYLOAD_MASS__KG_"]

# Summary Statistics
xbar2019 = np.mean(data2019)
xbar2020 = np.mean(data2020)
median2019 = np.median(data2019)
median2020 = np.median(data2020)
stdd2019 = np.std(data2019, ddof=1)
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


# Box and Whisker
plt.boxplot([data2019, data2020], labels=["2019", "2020"])
plt.ylabel("Payload Mass (kg)"); plt.legend()
plt.savefig("SpaceX Payload Mass Boxplots")

# Confidence Interval

# Hypothesis Test