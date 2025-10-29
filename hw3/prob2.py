import numpy as np


# Forward model
def H(x):
    return np.exp(x[0] * x[1]) + np.log(1 + x[2]**2)

def H_jacobian(x):
    return np.array([x[1] * np.exp(x[0] * x[1]), x[0] * np.exp(x[0] * x[1]), 2 * x[2] / (1 + x[2]**2)])

# H_TLM: The code that commputes delta_y given jacob of H evalulated at x, then @ delta_x
def H_TLM(x, delta_x):
    return H_jacobian(x).T @ delta_x

# Tangent Linear, the full numerator over denominator function 
def tangent_test(x, delta_x):
    return (H(x + delta_x) - H(x)) / H_TLM(x, delta_x)

# H_adj: The code that computes delta_x as H.T @ delta_y using H.T evaluated at x
def H_adjoint(x, delta_y):
    return H_jacobian(x) @ delta_y

# Adjoint, returns the error between LHS and RHS of the adjoint equation
# Where delta_y is obtained from H_TLM(x, delta_x)
def adjoint_test(x, delta_x):
    LHS = H_TLM(x, delta_x).T @ H_TLM(x, delta_x)
    delta_y = H_TLM(x, delta_x)
    RHS = delta_x.T @ H_adjoint(x, delta_y)
    return LHS - RHS


x0 = np.array([-1.0, -1.0, -1.0]).reshape(3, 1)
delta_x = np.array([0.01, 0.01, 0.01]).reshape(3, 1)

print(f"x0: {x0.T}")
print(f"Forwards Model: {H(x0)}")
print(f"Tangent Test: {tangent_test(x0, delta_x)}")
print(f"Adjoint Test: {adjoint_test(x0, delta_x)}")