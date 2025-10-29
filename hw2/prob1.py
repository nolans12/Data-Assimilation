# Goal is to guess the locatino of the car in 2D [East-West, North-South]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as patches

# You have 2 measurements
ya = np.array([1, 0])  # [East-West, North-South] in km
# Ra = np.array([[0.6, -0.6], [-0.6, 1.2]])
Ra = np.array([[0.6, 0], [0, 1.2]])
yb = np.array([4, 0])  # [East-West, North-South] in km
# Rb = np.array([[0.6, 0.6], [0.6, 1.2]])
Rb = np.array([[0.6, 0], [0, 1.2]])

# Solve w/ MLE
x_MLE = np.linalg.inv(np.linalg.inv(Ra) + np.linalg.inv(Rb)) @ (np.linalg.inv(Ra) @ ya + np.linalg.inv(Rb) @ yb)

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

# Plot MLE
ax.plot(x_MLE[0], x_MLE[1], 'go', markersize=8, label='MLE', zorder=3)

# Set up the plot
ax.set_xlabel('East-West [km]', fontsize=12)
ax.set_ylabel('North-South [km]', fontsize=12)
ax.set_title('Car Location Measurements with Uncertainty Ellipses', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_aspect('equal')

# Set reasonable axis limits
margin = 2
ax.set_xlim(-2, 6)
ax.set_ylim(-4, 4)

plt.tight_layout()
plt.show()