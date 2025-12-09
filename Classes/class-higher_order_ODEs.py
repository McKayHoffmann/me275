# See notes for equation derivation
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def spring_system(t, y):
    parameters = np.array([9.81, 80.0, 10.0, 20.0])
    g, m, k, c = parameters
    x, v = y
    dxdt = v    # solve_ivp returns the integral of these two things
    dvdt = (m*g - c*v - k*x) / m
    return np.array([dxdt, dvdt])

parameters = np.array([9.81, 80.0, 10.0, 20.0])
init_cond = np.array([100, 0])

t_eval = np.linspace(0, 20, 500)
sol = solve_ivp(spring_system, (0, 20), init_cond, t_eval=t_eval)
position = sol.y[0]
velocity = sol.y[1]
time = sol.t
plt.plot(time, position, label="Position")
plt.plot(time, velocity, label="Velocity")
plt.legend()

# Personal example
def harmonic_spring(t, y):
    m = 10
    gamma = 1
    k = 10
    x, v = y
    dydt = v
    dvdt = (-gamma*v - k*x) / m
    return np.array([dydt, dvdt])
accel = lambda x, v: (-1*v - 10*x) / 10
init_cond = [10, 0]

t_eval = np.linspace(0, 50, 500)
sol = solve_ivp(harmonic_spring, (0, 50), init_cond, t_eval=t_eval)
time = sol.t
position = sol.y[0]
velocity = sol.y[1]
acceleration = np.array([accel(xi, vi) for (xi, vi) in zip(position, velocity)])
plt.plot(time, position, label="Position")
plt.plot(time, velocity, label="Velocity")
plt.plot(time, acceleration, label="Acceleration")
plt.legend()

# DAY 2, PART 2
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def simple_motor(t, y, params):
    w, i = y
    Vs, R, L, Kb, K1, J, b = params
    dwdt = (K1*i - b*w) / (J)
    didt = (Vs - Kb*w - R*i) / (L)
    return dwdt, didt

params = 12, 2, 0.5, 0.1, 0.1, 0.01, 0.02
init_cond = np.array([10, 5])
myTime = np.linspace(0, 10, 500)
sol = solve_ivp(simple_motor, (0, 10), init_cond, t_eval=myTime, args=(params,))
w_values = sol.y[0]
i_values = sol.y[1]

plt.plot(myTime, w_values, label="Omega")
plt.plot(myTime, i_values, label="i")
plt.legend()

# Non-linear pendulum

def pendulum_ode(t, y):
    theta, thetadot = y
    dthetadt = thetadot
    dthetadotdt = 
    return [dthetadt, dthetadotdt[]

## BOUNDARY VALUE PROBLEMS
