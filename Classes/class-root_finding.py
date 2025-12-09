# When I have a variable that I can't isolate to solve for... residual equation
# Residual equation
# r(m) = 0

# Step 0: Find or be given a bracket [a, b] --> f(a)*f(b) < 0
# Step 1: Evaluate at midpoint --> r((a + b)/2)
# Step 2: Is r((a + b) / 2) positive or negative? Redefine bracket
# Step 3: Repeat until it approaches 0

# Multiple ways to import a function from a package
import scipy
# scipy.optimize.root_scalar(f)

from scipy import optimize as op
# op.root_scalar(f)

from scipy.optimize import root_scalar

# -- BRACKETING METHOD -- #

f = lambda x: x**3 - 1  # Residual I want to solve
root_scalar(f, bracket=(-3, 8), method='bisect')    # Iterations: 43
root_scalar(f, bracket=(0, 2), method='bisect')    # Iterations: 1
root_scalar(f, bracket=(0, 3), method='bisect')    # Iterations: 41
root_scalar(f, bracket=(-5, -2), method='bisect')   # Doesnt' work!

# -- NEWTON'S METHOD -- #
# Instead of using the midpoint, we are going to make guesses using derivitives
# Step 1: Make a guess, evaluate the function, and take the derivative

# -- IN CLASS PRACTICE PROBLEMS -- #
import math
g = lambda x: x - math.exp(-x)
g(0.5)
g(0.75)
root_scalar(g, bracket=(0.5, 0.75), method='bisect')    # iterations: 37
root_scalar(g, method='newton', x0=1)   # iterations: 4
root_scalar(g, bracket=(0.5, 0.75), method='brentq')    # iterations: 6

h = lambda gamma, alpha, Beta: (Beta) / (gamma + alpha) - math.tan(gamma)

def make_function(alpha, Beta):     # Closure function
    h = lambda gamma: (Beta) / (gamma + alpha) - math.tan(gamma)
    return h

h = lambda gamma, alpha=0.2, Beta=1.5: (Beta) / (gamma + alpha) - math.tan(gamma)

k = make_function(0.2, 1.5)     # Use closure function to define alpha and beta

# Other option
def beam(gamma, alpha, beta):
    return np.tan(gamma) - beta / (gamma + alpha)

alpha = 0.2
beta = 1.5
def residual(gamma):
    return beam(gamma, alpha, beta)

k(1)
k(0.9)  # Finding bracket
root_scalar(k, bracket=(1, 0.9), method='bisect')    # iterations: 36
root_scalar(k, x0=1, method='newton')   # iterations: 4
root_scalar(k, bracket=(1, 0.9), method='brentq')   # iterations: 6


