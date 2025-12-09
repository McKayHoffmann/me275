# HW 7
# ME EN 275
# Author: McKay Hoffmann
# Date: 10/30/25
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# -- Problem 1 -- #
def mssprgdamp(t, y, parameters):
    x, xdot = y
    m, b, k = parameters
    xdot = xdot
    xddot = (-b*xdot - k*x) / m
    return xdot, xddot

parameters1 = (20, 5, 20)   # m, b, k
parameters2 = (20, 40, 20)   # m, b, k
parameters3 = (20, 200, 20)   # m, b, k
myTime = np.arange(0, 15, 0.1)
sol1 = solve_ivp(mssprgdamp, (0, 15), [1, 0], t_eval=myTime, args=(parameters1,))
sol2 = solve_ivp(mssprgdamp, (0, 15), [1, 0], t_eval=myTime, args=(parameters2,))
sol3 = solve_ivp(mssprgdamp, (0, 15), [1, 0], t_eval=myTime, args=(parameters3,))
pos1 = sol1.y[0]
pos2 = sol2.y[0]
pos3 = sol3.y[0]

plt.plot(myTime, pos1, label="b = 5")
plt.plot(myTime, pos2, label="b = 40")
plt.plot(myTime, pos3, label="b = 200")
plt.legend(); plt.xlabel("Time $(s)$"); plt.ylabel("Position")
plt.savefig("Mass Spring Dampener")

# -- Problem 2 -- #
# Scenario 1
def SIRModel(t, y, parameters):
    S, I, R = y     # Susceptible, Infected, and Recovered
    i, r = parameters   # infection rate, recovery rate
    Sdot = -i*S*I
    Idot = i*S*I - r*I
    Rdot = r*I
    return Sdot, Idot, Rdot

def event_infected_reaches_10(t, y, parameters):
    S, I, R = y
    return I - 10

event_infected_reaches_10.terminal = True
event_infected_reaches_10.direction = -1

myTime = np.arange(0, 100, 0.1)
SIRparameters = (0.002/7, 0.15)
SIRsol = solve_ivp(SIRModel, (0, 100), [10000, 1, 0], t_eval=myTime, args=(SIRparameters,), events=event_infected_reaches_10)
susceptible = SIRsol.y[0]
infected = SIRsol.y[1]
recovered = SIRsol.y[2]
event_time = SIRsol.t_events[0][0]  # 49.52 days

plt.plot(SIRsol.t, susceptible, label="Susceptible")
plt.plot(SIRsol.t, infected, label="Infected")
plt.plot(SIRsol.t, recovered, label="Recovered")
plt.xlabel("Time $(days)$"); plt.ylabel("Persons"); plt.legend()
plt.savefig("SIR Model Scenario 1.png")

# Scenario 2
def SIRModel2(t, y, parameters):
    S, I, R = y     # Susceptible, Infected, and Recovered
    i, r, p = parameters   # infection rate, recovery rate, return rate
    Sdot = -i*S*I + p*R
    Idot = i*S*I - r*I
    Rdot = r*I - p*R
    return Sdot, Idot, Rdot

SIRparameters2 = (0.002/7, 0.15, 0.03)
myTime = np.arange(0, 100, 0.1)
SIRsol2 = solve_ivp(SIRModel2, (0, 100), [10000, 1, 0], t_eval=myTime, args=(SIRparameters2,), events=event_infected_reaches_10)
susceptible2 = SIRsol2.y[0]
infected2 = SIRsol2.y[1]
recovered2 = SIRsol2.y[2]
# event_time2 = SIRsol2.t_events[0][0]  # Never?

plt.plot(SIRsol2.t, susceptible2, label="Susceptible")
plt.plot(SIRsol2.t, infected2, label="Infected")
plt.plot(SIRsol2.t, recovered2, label="Recovered")
plt.xlabel("Time $(days)$"); plt.ylabel("Persons"); plt.legend()
plt.savefig("SIR Model Scenario 2.png")

# -- Problem 3 -- #
N = 100
occupants = np.array([1, 2, 3, 4, 5])
num_cars = np.array([70, 15, 10, 3, 2])
data = np.repeat(occupants, num_cars)

# a
mean = np.mean(data)    # 1.52
# b
stdev = np.std(data)    # 0.9325234581499813
# c
median = np.median(data)    # 1.0
# d
first_qrtl = stats.scoreatpercentile(data, per=25)  # 1.0
third_qrtl = stats.scoreatpercentile(data, per=75)  # 2.0
# e
1 - stats.percentileofscore(data, mean)/100     # .30
(15+10+3+2)/100