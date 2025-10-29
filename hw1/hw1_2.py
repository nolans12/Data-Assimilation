import numpy as np

# Prior mean and covariance
x_prior = np.array([2.3, 2.5])    
P_prior = np.array([[0.225, 0.0],[0.0, 0.15]])

# Measurement
y = 3.0
R = np.array([[0.15]])     

# Observation mat - only observe x1
H = np.array([[1.0, 0.0]])       

# Innovation z
z = y - H @ x_prior

# Kalman gain
K = P_prior @ H.T @ np.linalg.inv(H @ P_prior @ H.T + R)

# Posterior mean
x_posterior = x_prior + (K @ z).flatten()

# Posterior covariance
P_posterior = (np.eye(2) - K @ H) @ P_prior

print("Posterior mean (xa):\n", x_posterior)
print("Posterior covariance (Ba):\n", P_posterior)
