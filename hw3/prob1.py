import jax.numpy as np
from jax import grad, jacfwd, jacrev, hessian
from scipy.optimize import minimize

# Given
x_b = np.array([1, -1, 2.5]).reshape(3, 1)
B = np.array([[0.1, 0.02, 0],
              [0.02, 0.1, 0.01],
              [0, 0.01, 0.2]])
B_inv = np.linalg.inv(B)

epsilon = 0.04
R = np.array([[epsilon**2]])  # observation error covariance
R_inv = np.linalg.inv(R)

# Measurement!
y_obs = 2

# Forward model - no noise, that is applied as R
def H(x):
    return np.exp(x[0] * x[1]) + np.log(1 + x[2]**2)

# Cost function
def J(x):
    hx = H(x)
    dy = y_obs - hx
    dx = x - x_b
    cost = 0.5 * dy.T @ R_inv @ dy + 0.5 * dx.T @ B_inv @ dx
    return np.squeeze(cost)

# Grad and Hessian
grad_J = grad(J)
hessian_J = hessian(J)

# Newtons
def newton_method(x0, tol=1e-6, max_iter=100):
    x = x0.copy()
    for i in range(max_iter):
        g = grad_J(x).reshape(3, 1)
        Hx = hessian_J(x).reshape(3, 3)
        x_new = x - np.linalg.inv(Hx) @ g
        if np.linalg.norm(x_new - x) < tol:
            print(f"Newton converged in {i+1} iterations.")
            return x_new
        x = x_new
    print("Newton did not fully converge.")
    return x


# Gradient Descent w/ alpha from quad
def steepest_descent(x0, tol=1e-6, max_iter=100):
    x = x0.copy()
    for i in range(max_iter):
        g = grad_J(x).reshape(3, 1)
        # use quad optimization
        alpha = g.T @ g / (g.T @ hessian_J(x).reshape(3, 3) @ g)
        print(f"Alpha: {alpha}")
        x_new = x - alpha * g
        if np.linalg.norm(x_new - x) < tol:
            print(f"Steepest Descent converged in {i+1} iterations.")
            return x_new
        x = x_new
    print("Steepest Descent did not fully converge.")
    return x

x0 = np.array([0.5, -0.5, 1.0]).reshape(3, 1)

x_newton = newton_method(x0)
x_sd = steepest_descent(x0)

print("\nResults:")
print("Newton’s method solution:\n", x_newton)
print("Steepest Descent solution:\n", x_sd)
print("Cost J(x) Newton:", J(x_newton))
print("Cost J(x) SD:", J(x_sd))
