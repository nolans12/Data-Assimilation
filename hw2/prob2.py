# Goal is to guess the locatino of the car in 2D [East-West, North-South]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as patches

# You have 2 measurements
ya = np.array([1, 0]).reshape(2,1)  # [East-West, North-South] in km
Ra = np.array([[0.6, -0.6], [-0.6, 1.2]])
yb = np.array([4, 0]).reshape(2,1)  # [East-West, North-South] in km
Rb = np.array([[0.6, 0.6], [0.6, 1.2]])

# Prior
x_prior = np.array([0, 0]).reshape(2,1)
P_prior = np.array([[1.2, 0.6], [0.6, 0.6]])

# bayesian approach
# P_posterior = np.linalg.inv(np.linalg.inv(P_prior) + np.linalg.inv(Ra) + np.linalg.inv(Rb))
# x_MAP = P_posterior @ (np.linalg.inv(P_prior) @ x_prior + 
#                        np.linalg.inv(Ra) @ ya + 
#                        np.linalg.inv(Rb) @ yb)


H = np.vstack([np.eye(2), np.eye(2)])
y = np.vstack([ya, yb])
R = np.zeros((4, 4))
R[:2, :2] = Ra
R[2:, 2:] = Rb
x_MAP = x_prior + P_prior @ H.T @ np.linalg.inv(H @ P_prior @ H.T + R) @ (y - H @ x_prior)


print("MAP estimate:", x_MAP.flatten())

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot measurement points
ax.plot(ya[0], ya[1], 'ro', markersize=8, label='ya measurement', zorder=3)
ax.plot(yb[0], yb[1], 'bo', markersize=8, label='yb measurement', zorder=3)

# Create confidence ellipses (95% confidence)
confidence_level = 0.95
chi2_val = 5.991  # Chi-squared value for 95% confidence with 2 DOF

# For ya measurement
eigenvals_a, eigenvecs_a = np.linalg.eigh(Ra)
angle_a = np.degrees(np.arctan2(eigenvecs_a[1, 0], eigenvecs_a[0, 0]))
width_a = 2 * np.sqrt(chi2_val * eigenvals_a[0])
height_a = 2 * np.sqrt(chi2_val * eigenvals_a[1])

ellipse_a = Ellipse((ya[0], ya[1]), width_a, height_a, angle=angle_a,
                    alpha=0.3, facecolor='red', edgecolor='red', linewidth=2,
                    label='ya uncertainty ellipse')
ax.add_patch(ellipse_a)

# For yb measurement
eigenvals_b, eigenvecs_b = np.linalg.eigh(Rb)
angle_b = np.degrees(np.arctan2(eigenvecs_b[1, 0], eigenvecs_b[0, 0]))
width_b = 2 * np.sqrt(chi2_val * eigenvals_b[0])
height_b = 2 * np.sqrt(chi2_val * eigenvals_b[1])

ellipse_b = Ellipse((yb[0], yb[1]), width_b, height_b, angle=angle_b,
                    alpha=0.3, facecolor='blue', edgecolor='blue', linewidth=2,
                    label='yb uncertainty ellipse')
ax.add_patch(ellipse_b)

# Prior point
ax.plot(x_prior[0,0], x_prior[1,0], 'ko', markersize=8, label='prior mean', zorder=3)

# Prior uncertainty ellipse (95%)
eigenvals_p, eigenvecs_p = np.linalg.eigh(P_prior)
angle_p = np.degrees(np.arctan2(eigenvecs_p[1, 0], eigenvecs_p[0, 0]))
width_p = 2 * np.sqrt(chi2_val * eigenvals_p[0])
height_p = 2 * np.sqrt(chi2_val * eigenvals_p[1])

ellipse_p = Ellipse((x_prior[0,0], x_prior[1,0]), width_p, height_p, angle=angle_p,
                    alpha=0.2, facecolor='gray', edgecolor='black',
                    linestyle='--', linewidth=2, label='prior uncertainty ellipse')
ax.add_patch(ellipse_p)

# Plot MAP estimate
ax.plot(x_MAP[0], x_MAP[1], 'g*', markersize=12, label='MAP estimate', zorder=3)

# # Plot posterior uncertainty ellipse
# eigenvals_post, eigenvecs_post = np.linalg.eigh(P_posterior)
# angle_post = np.degrees(np.arctan2(eigenvecs_post[1, 0], eigenvecs_post[0, 0]))
# width_post = 2 * np.sqrt(chi2_val * eigenvals_post[0])
# height_post = 2 * np.sqrt(chi2_val * eigenvals_post[1])

# ellipse_post = Ellipse((x_MAP[0], x_MAP[1]), width_post, height_post, angle=angle_post,
#                        alpha=0.3, facecolor='green', edgecolor='green', linewidth=2,
#                        label='posterior uncertainty ellipse')
# ax.add_patch(ellipse_post)

# Set up the plot
ax.set_xlabel('East-West [km]', fontsize=12)
ax.set_ylabel('North-South [km]', fontsize=12)
ax.set_title('MAP Estimation with Prior and Two Measurements', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_aspect('equal')

# Set reasonable axis limits
ax.set_xlim(-2, 6)
ax.set_ylim(-4, 4)

plt.tight_layout()
plt.show()