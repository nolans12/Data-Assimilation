import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_cov_ellipse(mean, cov, ax, nstd=2.0, **kwargs):
    """Plot an ellipse to represent the covariance of a Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height,
                      angle=theta, **kwargs)
    ax.add_patch(ellipse)

# Kalman update 
def bayes_update(xb, B, H, y, R):
    K = B @ H.T @ np.linalg.inv(H @ B @ H.T + R)
    d = y - H @ xb
    xa = xb + (K @ d).flatten()
    Ba = (np.eye(len(xb)) - K @ H) @ B
    return xa, Ba, K

xb = np.array([2.3, 2.5])
y = 3.0
H = np.array([[1.0, 0.0]])
R = np.array([[0.15]])

B1 = np.array([[0.225, 0.05],
               [0.05, 0.15]])
xa1, Ba1, K1 = bayes_update(xb, B1, H, y, R)

B2 = np.array([[0.225, 0.0],
               [0.0, 0.15]])
xa2, Ba2, K2 = bayes_update(xb, B2, H, y, R)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

ax = axes[0]
ax.set_title("Case 1: Correlated Prior")
plot_cov_ellipse(xb, B1, ax, edgecolor='blue', alpha=0.3, label="Prior")
plot_cov_ellipse(xa1, Ba1, ax, edgecolor='red', alpha=0.6, label="Posterior")
ax.scatter(*xb, c='blue', marker='x', label="Prior mean")
ax.scatter(*xa1, c='red', marker='o', label="Posterior mean")
ax.axvline(x=y, color='black', linestyle='--', label="Observation")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend(); ax.grid(True)

ax = axes[1]
ax.set_title("Case 2: Uncorrelated Prior")
plot_cov_ellipse(xb, B2, ax, edgecolor='blue', alpha=0.3, label="Prior")
plot_cov_ellipse(xa2, Ba2, ax, edgecolor='red', alpha=0.6, label="Posterior")
ax.scatter(*xb, c='blue', marker='x', label="Prior mean")
ax.scatter(*xa2, c='red', marker='o', label="Posterior mean")
ax.axvline(x=y, color='black', linestyle='--', label="Observation")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend(); ax.grid(True)

plt.tight_layout()
plt.show()
