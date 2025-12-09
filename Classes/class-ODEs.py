# Class - ODEs
# Date: 10/9/2025

### Part 1
## First order ODES
# Need:
# 1. First order ODE
# 2. Initial/boundary condition
# 3. Range (a < t < b)

# Euler's method: y_{i+1} = y_i + h * f(xi, yi)

### Part 2
## Improved Euler: Prediction corrector method
# Instead of just using the first derivative and trusting it, take a hypothetical step with a first derivative, find the next hypothetical derivative, go back and 
# average the two derivatives, then take a real step forward.

# IVP Example
import scipy
import matplotlib.pyplot as plt
import numpy as np

def odefun(t,y):
    return 0.1 * y

results = scipy.integrate.solve_ivp(odefun, (0, 100), [1])
sol = results.y[0, :]   # In the future, there will be several arrays for each of the outputs (multivariable ODEs)
len(results.y[0])
plt.plot(results.t, sol)

# How to get more points
mytime = np.linspace(0, 100, 1000)
results2 = scipy.integrate.solve_ivp(odefun, (0, 100), [1], t_eval=mytime)
sol2 = results2.y[0, :]   # In the future, there will be several arrays for each of the outputs (multivariable ODEs)
plt.plot(results.t, sol, "o")   # Points chosen by adaptive time stepping
plt.plot(results2.t, sol2)      # Trajectory I want to see

results3 = scipy.integrate.solve_ivp(odefun, (0, 100), [1], dense_output=True)
sol3 = results3.y[0, :]   # In the future, there will be several arrays for each of the outputs (multivariable ODEs)
plt.plot(results3.t, sol3)
len(results3.y[0])

# Another example
def odefun2(x, y):
    return 2 * x * y

myX = np.arange(0, 3, 0.001)
solution = scipy.integrate.solve_ivp(odefun2, (0, 3), [1], t_eval=myX)  # Function, range, initial conditions, points I want the y values for
plt.plot(solution.t, solution.y[0])

## System of ODEs
# If we have 3 First Order ODEs we want to solve simutaneously, we also need to provide 3 initial conditions

# Predator Prey Example (Coupled First-Order ODE)
# States: [u, v] u: prey, v: predator
# du/dt = alpha * u - beta * v
# dv/t = -gamma * v + delta * u * v
# alpha = 1, beta = 0.01, gamma = 1, delta = 0.02
# time goes from 0 to 15
# u(0) = 40, v(0) = 10
import numpy as np
import scipy
import matplotlib.pyplot as plt

alpha = 1
beta = 0.01
gamma = 1
delta = 0.02

def predator_prey_rates(t, y):  # Note, this is how EVERY ODE PROBLEM NEEDS TO BE FORMATTED
    u, v = y
    dudt = alpha*u - beta*u*v
    dvdt = -gamma*v + delta*u*v
    return np.array([dudt, dvdt])

init_conditions = np.array([40, 10])

myTime = np.linspace(0, 15, 200)
sol = scipy.integrate.solve_ivp(predator_prey_rates, (0, 15), init_conditions, t_eval=myTime)
time = sol.t
rabbit_pop = sol.y[0]
wolf_pop = sol.y[1]

plt.plot(time, rabbit_pop, label="Rabbit Population")
plt.plot(time, wolf_pop, label="Wolf Population")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()