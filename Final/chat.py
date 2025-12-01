"""
bias_field_live_sim_both.py
----------------------------------------------------------
Live simulation of GLOBAL BIAS ESTIMATION (Az + El) using:
  - A parametric bias model (11 params)
  - Sequential 3D-Var updates
  - 50 SIRE ground sites
  - Noisy “measurements” at each timestep
  - Live Cartopy world map (two subplots)
----------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path


# ============================================================
#                  PARAMETRIC BIAS FIELD MODEL
# ============================================================

TRUE_THETA = np.deg2rad([
    -0.2, 0.1, 0.2, -0.15, -0.05, 0.3,   # Azimuth params (6)
    -0.03, 0.01, 0.03, 0.03, 0.01        # Elevation params (5)
])

def bias_model(lat_deg, lon_deg, theta=TRUE_THETA):
    """Return (Az_bias_deg, El_bias_deg) on a lat/lon grid."""
    phi, lam = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    a0,a1,a2,a3,a4,a5, e0,e1,e2,e3,e4 = theta

    # Azimuth bias
    bA = (a0 + a1*np.sin(phi)
             + a2*np.cos(phi)*np.cos(lam)
             + a3*np.cos(phi)*np.sin(lam)
             + a4*np.sin(2*phi)*np.cos(2*lam)
             + a5*np.sin(2*phi)*np.sin(2*lam))

    # Elevation bias
    bE = (e0 + e1*0.5*(3*np.sin(phi)**2 - 1)
              + e2*np.cos(phi)*np.cos(lam)
              - e3*np.cos(phi)*np.sin(lam)
              + e4*np.sin(2*phi))

    return np.rad2deg(bA), np.rad2deg(bE)



# ============================================================
#                 BASIS FUNCTIONS (H MATRIX)
# ============================================================

def h_alpha(lat, lon):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([
        np.ones_like(phi),
        np.sin(phi),
        np.cos(phi)*np.cos(lam),
        np.cos(phi)*np.sin(lam),
        np.sin(2*phi)*np.cos(2*lam),
        np.sin(2*phi)*np.sin(2*lam),
    ], axis=-1)   # shape: (N,6)

def h_elev(lat, lon):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([
        np.ones_like(phi),
        0.5*(3*np.sin(phi)**2 - 1),
        np.cos(phi)*np.cos(lam),
        -np.cos(phi)*np.sin(lam),
        np.sin(2*phi),
    ], axis=-1)  # shape: (N,5)



# ============================================================
#                    3D-VAR UPDATE
# ============================================================

def three_d_var_update(theta_b, B, H, y, R):
    HBHT = H @ B @ H.T
    S = HBHT + R
    K = B @ H.T @ np.linalg.inv(S)
    innov = y - H @ theta_b
    theta_a = theta_b + K @ innov

    I = np.eye(B.shape[0])
    B_a = (I - K @ H) @ B @ (I - K @ H).T + K @ R @ K.T
    return theta_a, B_a



# ============================================================
#             UNIFORM SPHERE SAMPLING FOR SIRE SITES
# ============================================================

def fibonacci_sites(n, seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i+0.5)/n) - np.pi/2
    lam = (2*np.pi/(np.sqrt(5)+1))*i + rng.uniform(-0.1,0.1,size=n)
    return np.rad2deg(phi), ((np.rad2deg(lam)+180)%360 - 180)



# ============================================================
#                   LIVE PLOTTER (2 PANELS)
# ============================================================

class BiasFieldLivePlotterBoth:
    """Side-by-side live visualization of True/Estimated (Az, El)."""

    def __init__(self, save_frames=True):
        plt.ion()
        self.fig = plt.figure(figsize=(18, 7))

        # Left = Azimuth field
        self.axA = self.fig.add_subplot(1,2,1,projection=ccrs.PlateCarree())
        self.axA.set_global()
        # Right = Elevation field
        self.axE = self.fig.add_subplot(1,2,2,projection=ccrs.PlateCarree())
        self.axE.set_global()

        for ax in (self.axA, self.axE):
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            ax.add_feature(cfeature.LAND, facecolor="#f4f2ec")
            ax.add_feature(cfeature.OCEAN, facecolor="#c6dbef")
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray",
                              alpha=0.5, linestyle="--")
            gl.top_labels = gl.right_labels = False

        # Grid for plotting
        self.lat = np.linspace(-90,90,181)
        self.lon = np.linspace(-180,180,361)
        self.LON, self.LAT = np.meshgrid(self.lon, self.lat)

        # Plot handles
        self.trueA = None
        self.trueE = None
        self.estA = None
        self.estE = None
        self.sitesA = None
        self.sitesE = None
        self.measA = None
        self.measE = None

        self.save_frames = save_frames
        self.frame_dir = Path("bias_frames")
        self.frame_dir.mkdir(exist_ok=True)
        self.frame_count = 0

    # -----------------------------------------------------------------

    def update(self, step, theta_est, sire_lat, sire_lon, measA_rad, measE_rad):

        # ======================================================
        # 1) CLEAR AXES COMPLETELY (RESET PLOTS)
        # ======================================================
        self.axA.clear()
        self.axE.clear()

        # ------------------------------------------------------
        # 1a) REBUILD BASE MAPS (coastlines, land, ocean, grid)
        # ------------------------------------------------------
        for ax in (self.axA, self.axE):
            ax.set_global()
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            ax.add_feature(cfeature.LAND, facecolor="#f4f2ec")
            ax.add_feature(cfeature.OCEAN, facecolor="#c6dbef")

            gl = ax.gridlines(
                draw_labels=True,
                linewidth=0.5,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False

        # ======================================================
        # 2) COMPUTE TRUE & ESTIMATED FIELDS
        # ======================================================
        BA_true, BE_true = bias_model(self.LAT, self.LON)
        BA_est, BE_est   = bias_model(self.LAT, self.LON, theta_est)

        # ======================================================
        # 3) DRAW TRUE FIELDS
        # ======================================================
        self.axA.pcolormesh(
            self.lon, self.lat, BA_true,
            cmap="coolwarm",
            shading="auto",
            alpha=0.55,
            transform=ccrs.PlateCarree(),
        )
        self.axE.pcolormesh(
            self.lon, self.lat, BE_true,
            cmap="coolwarm",
            shading="auto",
            alpha=0.55,
            transform=ccrs.PlateCarree(),
        )

        # ======================================================
        # 4) DRAW ESTIMATED FIELDS (contours)
        # ======================================================
        self.axA.contour(
            self.lon, self.lat, BA_est,
            levels=12,
            colors="black",
            linewidths=1,
            transform=ccrs.PlateCarree(),
        )

        self.axE.contour(
            self.lon, self.lat, BE_est,
            levels=12,
            colors="black",
            linewidths=1,
            transform=ccrs.PlateCarree(),
        )

        # ======================================================
        # 5) SIRE SITES + MEASUREMENTS
        # ======================================================
        # Plot SIRE station locations
        for ax in (self.axA, self.axE):
            ax.scatter(
                sire_lon, sire_lat,
                s=40,
                c="yellow",
                edgecolors="black",
                transform=ccrs.PlateCarree(),
                zorder=5,
                label="SIRE Sites",
            )

            ax.scatter(
                sire_lon, sire_lat,
                s=30,
                c="red",
                marker="x",
                alpha=0.6,
                transform=ccrs.PlateCarree(),
                zorder=6,
                label="Measurements",
            )

        # ======================================================
        # 6) TITLES
        # ======================================================
        self.axA.set_title(f"Azimuth Bias Field — Step {step}", fontsize=13, weight="bold")
        self.axE.set_title(f"Elevation Bias Field — Step {step}", fontsize=13, weight="bold")

        # ======================================================
        # 7) RENDER + SAVE FRAME
        # ======================================================
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

        if self.save_frames:
            out = self.frame_dir / f"frame_{self.frame_count:04d}.png"
            self.fig.savefig(out, dpi=120)
            self.frame_count += 1


    # -----------------------------------------------------------------

    def close(self):
        plt.ioff()
        plt.show()



# ============================================================
#                    MAIN SIMULATION
# ============================================================

if __name__ == "__main__":

    N_SITES = 50
    N_STEPS = 30
    SIGMA = np.deg2rad(0.1)   # noise std (radians)

    # Generate SIRE sites
    sire_lat, sire_lon = fibonacci_sites(N_SITES)

    # Build full H matrix (2N × 11)
    Halpha = h_alpha(sire_lat, sire_lon)       # (50 × 6)
    Helev  = h_elev (sire_lat, sire_lon)       # (50 × 5)
    H = np.block([
        [Halpha, np.zeros((N_SITES, 5))],
        [np.zeros((N_SITES, 6)), Helev]
    ])

    # Full covariance R (2N × 2N)
    R = np.diag(np.full(2*N_SITES, SIGMA**2))

    # Prior (11-D)
    theta_est = np.zeros(11)
    B = np.diag(np.full(11, np.deg2rad(0.4)**2))

    # Live plotter
    plotter = BiasFieldLivePlotterBoth(save_frames=True)

    # Sequential updates
    for k in range(N_STEPS):

        # Sample TRUE bias at SIRE sites
        bA, bE = bias_model(sire_lat, sire_lon)
        yA = np.deg2rad(bA) + np.random.normal(0, SIGMA, N_SITES)
        yE = np.deg2rad(bE) + np.random.normal(0, SIGMA, N_SITES)

        # Stack into vector y (2N)
        y = np.concatenate([yA, yE])

        # 3D-Var update
        theta_est, B = three_d_var_update(theta_est, B, H, y, R)

        # Live update
        plotter.update(
            step=k,
            theta_est=theta_est,
            sire_lat=sire_lat,
            sire_lon=sire_lon,
            measA_rad=yA,
            measE_rad=yE,
        )

    plotter.close()
