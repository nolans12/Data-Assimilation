# Problem 3: Kalman Filter for tracking a car using constant position model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

SHOW_MEASUREMENT_NOISE = False 

# Assume constant position model
F = np.array([[1, 0],
              [0, 1]])

Q = np.array([[0.04, 0],
              [0, 0.04]])

H = np.array([[1, 0],
              [0, 1]])

Ra = np.array([[0.6, -0.6], [-0.6, 1.2]])
Rb = np.array([[0.6, 0.6], [0.6, 1.2]])

ya_measurements = np.array([[1, 1, 1, 1, 1, 1, 1],
                           [0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.5]])

yb_measurements = np.array([[4, 4, 4, 4, 4, 4, 4],
                           [0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.5]])

# Prior state
x0 = np.array([0, 0]).reshape(2,1)
P0 = np.array([[1.2, 0.6], [0.6, 0.6]])


n_steps = ya_measurements.shape[1]
x_estimates = np.zeros((2, n_steps))
P_estimates = np.zeros((2, 2, n_steps))
x_predictions = np.zeros((2, n_steps))
P_predictions = np.zeros((2, 2, n_steps))

# Init
x_est = x0.copy()
P_est = P0.copy()

# # True Position by 7.2 km/hr in south east
# dir_vec = np.array([1.0, -1.0])
# dir_vec = dir_vec / np.linalg.norm(dir_vec)
# init_pos = np.array([2.5, 0.0])
# true_positions = np.zeros((2, n_steps))
# for k in range(n_steps):
#     true_positions[:, k] = dir_vec * k * 7.2 / 60 + init_pos

# Kalman Filter Loop
for k in range(n_steps):
    # PREDICT
    x_pred = F @ x_est
    P_pred = F @ P_est @ F.T + Q

    x_predictions[:, k] = x_pred.flatten()
    P_predictions[:, :, k] = P_pred
    
    # UPDATE
    ya_k = ya_measurements[:, k].reshape(2, 1)
    yb_k = yb_measurements[:, k].reshape(2, 1)
    
    # Update w/ GPS A
    y_a_residual = ya_k - H @ x_pred  # innovation/residual
    S_a = H @ P_pred @ H.T + Ra       # innovation covariance
    K_a = P_pred @ H.T @ np.linalg.inv(S_a)  # Kalman gain
    
    x_temp = x_pred + K_a @ y_a_residual
    P_temp = (np.eye(2) - K_a @ H) @ P_pred
    
    # Update w/ GPS B
    y_b_residual = yb_k - H @ x_temp  # innovation/residual
    S_b = H @ P_temp @ H.T + Rb       # innovation covariance
    K_b = P_temp @ H.T @ np.linalg.inv(S_b)  # Kalman gain
    
    x_est = x_temp + K_b @ y_b_residual
    P_est = (np.eye(2) - K_b @ H) @ P_temp

    print(f"x_est: {x_est.flatten()}")
    print(f"P_est: {P_est}")

    # Store estimates
    x_estimates[:, k] = x_est.flatten()
    P_estimates[:, :, k] = P_est

# Visualization - Plot true position, estimated position, and covariance ellipses
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# # Plot true position trajectory
# ax.plot(true_positions[0, :], true_positions[1, :], 'k-', linewidth=3, 
#         label='True position', zorder=5)
# ax.plot(true_positions[0, :], true_positions[1, :], 'ko', markersize=8, zorder=5)

# Plot Kalman filter estimates
ax.plot(x_estimates[0, :], x_estimates[1, :], 'g-o', linewidth=2, markersize=6, 
        label='Kalman filter estimates', zorder=4)

# Plot GPS measurements
ax.plot(ya_measurements[0, :], ya_measurements[1, :], 'ro', markersize=6, 
        alpha=0.7, label='GPS A measurements', zorder=3)
ax.plot(yb_measurements[0, :], yb_measurements[1, :], 'bo', markersize=6, 
        alpha=0.7, label='GPS B measurements', zorder=3)


chi2_val = 5.991  # Chi-squared value for 95% confidence with 2 DOF

ellipse_legend = Ellipse((0, 0), 0, 0, alpha=0.8, facecolor='green', 
                        edgecolor='green', linewidth=1, label='Estimate covariances')
ax.add_patch(ellipse_legend)

# Add uncertainty ellipses for each estimate
for k in range(n_steps):
    P_k = P_estimates[:, :, k]
    eigenvals, eigenvecs = np.linalg.eigh(P_k)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * np.sqrt(chi2_val * eigenvals[0])
    height = 2 * np.sqrt(chi2_val * eigenvals[1])
    
    ellipse = Ellipse((x_estimates[0, k], x_estimates[1, k]), width, height, angle=angle,
                      alpha=0.2, facecolor='green', edgecolor='green', linewidth=1,
                      zorder=1)
    ax.add_patch(ellipse)


# Add measurement noise covariance ellipses at measurement locations (if enabled)
if SHOW_MEASUREMENT_NOISE:
    # Ra ellipses at GPS A locations
    for k in range(n_steps):
        eigenvals_ra, eigenvecs_ra = np.linalg.eigh(Ra)
        angle_ra = np.degrees(np.arctan2(eigenvecs_ra[1, 0], eigenvecs_ra[0, 0]))
        width_ra = 2 * np.sqrt(chi2_val * eigenvals_ra[0])
        height_ra = 2 * np.sqrt(chi2_val * eigenvals_ra[1])
        
        ellipse_ra = Ellipse((ya_measurements[0, k], ya_measurements[1, k]), 
                            width_ra, height_ra, angle=angle_ra,
                            alpha=0.15, facecolor='red', edgecolor='red', linewidth=1,
                            zorder=0)
        ax.add_patch(ellipse_ra)

    # Rb ellipses at GPS B locations  
    for k in range(n_steps):
        eigenvals_rb, eigenvecs_rb = np.linalg.eigh(Rb)
        angle_rb = np.degrees(np.arctan2(eigenvecs_rb[1, 0], eigenvecs_rb[0, 0]))
        width_rb = 2 * np.sqrt(chi2_val * eigenvals_rb[0])
        height_rb = 2 * np.sqrt(chi2_val * eigenvals_rb[1])
        
        ellipse_rb = Ellipse((yb_measurements[0, k], yb_measurements[1, k]), 
                            width_rb, height_rb, angle=angle_rb,
                            alpha=0.15, facecolor='blue', edgecolor='blue', linewidth=1,
                            zorder=0)
        ax.add_patch(ellipse_rb)

if SHOW_MEASUREMENT_NOISE:
    ellipse_ra_legend = Ellipse((0, 0), 0, 0, alpha=0.1, facecolor='red', 
                               edgecolor='red', linewidth=1, label='GPS A noise (Ra)')
    ax.add_patch(ellipse_ra_legend)

    ellipse_rb_legend = Ellipse((0, 0), 0, 0, alpha=0.1, facecolor='blue', 
                               edgecolor='blue', linewidth=1, label='GPS B noise (Rb)')
    ax.add_patch(ellipse_rb_legend)

ax.set_xlabel('East-West [km]')
ax.set_ylabel('North-South [km]')
ax.set_title('Kalman Filter: Estimated Position with Covariance Ellipses')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.show()


# Probability of [2, -2]
prob = (1 / (2 * np.pi * np.sqrt(np.linalg.det(P_estimates[:, :, -1])))) * np.exp(-0.5 * (np.array([2, -2]) - x_estimates[:, -1]) @ np.linalg.inv(P_estimates[:, :, -1]) @ (np.array([2, -2]) - x_estimates[:, -1]))
print(f"Probability of [2, -2]: {prob}")
