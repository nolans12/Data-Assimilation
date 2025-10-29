import numpy as np
import matplotlib.pyplot as plt

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

# Forward model
def H(x):
    return np.exp(x[0] * x[1]) + np.log(1 + x[2]**2)

# Cost function
def J(x):
    hx = H(x)
    dy = np.array([y_obs - hx]).reshape(1, 1)
    dx = x - x_b
    cost = 0.5 * dy.T @ R_inv @ dy + 0.5 * dx.T @ B_inv @ dx
    return np.squeeze(cost)

# Grid setup
x0_vals = np.linspace(0.8, 1.3, 100)
x1_vals = np.linspace(-1.4, -0.8, 100)
x2_fixed = 2.1

X0, X1 = np.meshgrid(x0_vals, x1_vals)
J_vals = np.zeros_like(X0)

# Evaluate cost function
for i in range(X0.shape[0]):
    for j in range(X0.shape[1]):
        x = np.array([X0[i, j], X1[i, j], x2_fixed]).reshape(3, 1)
        J_vals[i, j] = J(x)

x_ideal = np.array([1.0685, -1.0862, 2.1032])

plt.figure(figsize=(7, 6))
contour = plt.contour(X0, X1, J_vals, levels=30, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.title(f"3DVar Cost Function Contours J(x₀, x₁) for x₂ = {x2_fixed}")
plt.xlabel("x₀")
plt.ylabel("x₁")
plt.plot(x_ideal[0], x_ideal[1], 'ro', label="Optimized x")
plt.legend()
plt.tight_layout()
plt.show()
