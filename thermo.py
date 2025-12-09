import matplotlib.pyplot as plt
import numpy as np

# Getting delta u data
temp = np.array([500, 520, 537]) - 459.67

u_data = np.array([85.20, 88.62, 91.53])
# F = R - 459.67


def gimmeu(T_in):
    T_in_R = T_in + 459.67
    if 500 <= T_in_R and T_in_R <= 520:
        return ((T_in_R - 500)/(520 - 500))*(88.62 - 85.20) + 85.20
    elif 520 < T_in_R and T_in_R <= 537:
        return ((T_in_R - 520)/(537 - 520))*(91.53 - 88.62) + 88.62

# Define Q_out for hour 0 to 8
def cost_0_8(T_in):
    qin =  (1/40)*4996.8*(T_in - 47)
    win = qin / 3
    cost1 = win * 8 * 0.0041
    return cost1

# Define Q_out for hour 8 to 9
def cost_8_9(T_in):
    qin =  (2738.775*(90.16 - gimmeu(T_in))) + ((4996.8/40) * ((69+T_in)/2 - 47))
    win = qin / 3
    cost2 = win * 1 * 0.0041
    return cost2

def total_cost(T_in):
    return cost_0_8(T_in) + cost_8_9(T_in)

t_range = np.arange(69, 47, -0.1)

cost_data = np.array([total_cost(ti) for ti in t_range])

plt.plot(t_range, cost_data)
plt.xlabel("Set Temperature (F)"), plt.ylabel("Cost ($)"), plt.legend()
plt.savefig("Thermo.png")