# bias_estimator.py
# -----------------------------------------------------------
# Global Bias Field Estimation (Az + El)
# Using parametric 11-parameter model + 3D-Var sequential update
# Integrated into main.py via bias_estimator.step(t)
# -----------------------------------------------------------

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

from bias_field import bias_model, TRUE_THETA


# ============================================================
# Basis Functions (H blocks)
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
    ], axis=-1)

def h_elev(lat, lon):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([
        np.ones_like(phi),
        0.5*(3*np.sin(phi)**2 - 1),
        np.cos(phi)*np.cos(lam),
        -np.cos(phi)*np.sin(lam),
        np.sin(2*phi),
    ], axis=-1)


# ============================================================
# 3D-Var Update
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
# Uniform SIRE site distribution
# ============================================================

def fibonacci_sites(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    phi = np.arccos(1 - 2*(idx + 0.5)/n) - np.pi/2
    lam = (2*np.pi/(np.sqrt(5)+1)) * idx + rng.uniform(-0.1,0.1,size=n)
    return np.rad2deg(phi), ((np.rad2deg(lam)+180)%360 - 180)


# ============================================================
# 2-panel Live Plotter
# ============================================================

class BiasFieldLivePlotterBoth:

    def __init__(self, save_frames=True):
        plt.ion()
        self.fig = plt.figure(figsize=(18, 14))
        
        # Frame saving for animation
        self.save_frames = save_frames
        self.frame_dir = Path(__file__).resolve().parent / "data" / "bias_frames"
        self.frame_count = 0
        if self.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            # Clean old frames
            for old_frame in self.frame_dir.glob("frame_*.png"):
                old_frame.unlink()

        # Create 2x2 subplot layout
        # Top row: Estimated bias fields
        self.axA_est = self.fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
        self.axE_est = self.fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
        
        # Bottom row: True bias fields (static reference)
        self.axA_true = self.fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
        self.axE_true = self.fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())

        for ax in (self.axA_est, self.axE_est, self.axA_true, self.axE_true):
            ax.set_global()
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor="#f4f2ec")
            ax.add_feature(cfeature.OCEAN, facecolor="#c6dbef")
            gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
            gl.top_labels = gl.right_labels = False

        # Grid for fields
        self.lat = np.linspace(-90, 90, 181)
        self.lon = np.linspace(-180, 180, 361)
        self.LON, self.LAT = np.meshgrid(self.lon, self.lat)
        
        # Compute and plot true bias fields (static, only once)
        BA_true, BE_true = bias_model(self.LAT, self.LON, TRUE_THETA)
        
        self.axA_true.contourf(self.lon, self.lat, BA_true, levels=15, cmap="coolwarm",
                               transform=ccrs.PlateCarree(), alpha=0.7)
        self.axA_true.set_title("Azimuth Bias Field (TRUTH)", fontsize=13, weight="bold")
        
        self.axE_true.contourf(self.lon, self.lat, BE_true, levels=15, cmap="coolwarm",
                               transform=ccrs.PlateCarree(), alpha=0.7)
        self.axE_true.set_title("Elevation Bias Field (TRUTH)", fontsize=13, weight="bold")

    def update(self, step, theta_est, sire_lat, sire_lon, measA_rad, measE_rad):

        # ======================================================
        # 1) CLEAR TOP ROW AXES (ESTIMATED FIELDS ONLY)
        # ======================================================
        self.axA_est.clear()
        self.axE_est.clear()

        # ------------------------------------------------------
        # 1a) REBUILD BASE MAPS FOR ESTIMATED FIELDS
        # ------------------------------------------------------
        for ax in (self.axA_est, self.axE_est):
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
        # 2) COMPUTE ESTIMATED FIELDS
        # ======================================================
        BA_est, BE_est = bias_model(self.LAT, self.LON, theta_est)

        # ======================================================
        # 3) DRAW ESTIMATED FIELDS (filled contours)
        # ======================================================
        self.axA_est.contourf(
            self.lon, self.lat, BA_est,
            levels=15,
            cmap="coolwarm",
            transform=ccrs.PlateCarree(),
            alpha=0.7,
        )

        self.axE_est.contourf(
            self.lon, self.lat, BE_est,
            levels=15,
            cmap="coolwarm",
            transform=ccrs.PlateCarree(),
            alpha=0.7,
        )

        # ======================================================
        # 4) SIRE SITES + MEASUREMENTS ON ESTIMATED FIELDS
        # ======================================================
        # Plot SIRE station locations
        for ax in (self.axA_est, self.axE_est):
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
        time_min = step / 60.0
        self.axA_est.set_title(f"Azimuth Bias Field - ESTIMATE (t={time_min:.2f} min)", fontsize=13, weight="bold")
        self.axE_est.set_title(f"Elevation Bias Field - ESTIMATE (t={time_min:.2f} min)", fontsize=13, weight="bold")

        # ======================================================
        # 7) RENDER + SAVE FRAME
        # ======================================================
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

        # Save frame for animation
        if self.save_frames:
            frame_path = self.frame_dir / f"frame_{self.frame_count:04d}.png"
            self.fig.set_size_inches(18, 10, forward=True)
            self.fig.savefig(frame_path, dpi=100, bbox_inches=None)
            self.frame_count += 1

    def close(self):
        plt.ioff()
        
        # Create animation from saved frames
        if self.save_frames and self.frame_count > 0:
            print(f"\nCreating bias field animation from {self.frame_count} frames...")
            output_dir = self.frame_dir.parent
            
            try:
                import imageio
                # Create GIF
                gif_path = output_dir / "bias_field_animation.gif"
                frames = []
                for i in range(self.frame_count):
                    frame_path = self.frame_dir / f"frame_{i:04d}.png"
                    if frame_path.exists():
                        frames.append(imageio.imread(frame_path))
                
                if frames:
                    imageio.mimsave(gif_path, frames, fps=10, loop=0)
                    print(f"Saved bias field GIF to {gif_path}")
                    
                    # Try to create MP4 if ffmpeg is available
                    try:
                        mp4_path = output_dir / "bias_field_animation.mp4"
                        imageio.mimsave(mp4_path, frames, fps=10, codec='libx264')
                        print(f"Saved bias field MP4 to {mp4_path}")
                    except Exception as e:
                        print(f"Could not create MP4 (ffmpeg may not be installed): {e}")
                
            except ImportError:
                print("imageio not installed. Install with 'pip install imageio' to create animations.")
            except Exception as e:
                print(f"Error creating bias field animation: {e}")
        
        plt.show()



# ============================================================
#                MAIN ESTIMATOR CLASS
# ============================================================

class BiasFieldEstimatorLive:
    """
    Create once, then integrate into main.py with:

        bias_estimator = BiasFieldEstimatorLive(NUM_SIRES, NOISE_SIGMA, SEED)

        # in main loop each dt:
        if ESTIMATE_BIAS:
            bias_estimator.step(t)

    """

    def __init__(
        self,
        NUM_SIRES: int,
        NOISE_SIGMA: float,
        SEED: int = 0,
        SAVE_FRAMES: bool = True,
    ):

        self.N = NUM_SIRES
        self.sigma = NOISE_SIGMA
        self.seed = SEED
        self.rng = np.random.default_rng(SEED)

        # SIREs
        self.sire_lat, self.sire_lon = fibonacci_sites(NUM_SIRES, seed=SEED)

        # Build H matrix
        Halpha = h_alpha(self.sire_lat, self.sire_lon)
        Helev  = h_elev(self.sire_lat, self.sire_lon)

        # (2N × 11)
        self.H = np.block([
            [Halpha, np.zeros((self.N, 5))],
            [np.zeros((self.N, 6)), Helev]
        ])

        # R
        self.R = np.diag(np.full(2*self.N, self.sigma**2))

        # Initial estimate
        self.theta = np.zeros(11)
        self.B = np.diag(np.full(11, np.deg2rad(0.4)**2))

        # live plot
        self.plotter = BiasFieldLivePlotterBoth(save_frames=SAVE_FRAMES)

        # step counter
        self.step_index = 0


    # -----------------------------------------------------------
    # PUBLIC METHOD CALLED FROM main.py EACH TIME STEP
    # -----------------------------------------------------------
    def step(self, sim_time: float):
        """
        Perform one 3D-Var update using noisy samples of TRUE bias field
        at the SIRE locations.
        """

        # TRUE field at SIREs
        BA_true, BE_true = bias_model(self.sire_lat, self.sire_lon)

        # sample noisy observations
        yA = np.deg2rad(BA_true) + self.rng.normal(0, self.sigma, size=self.N)
        yE = np.deg2rad(BE_true) + self.rng.normal(0, self.sigma, size=self.N)
        y = np.concatenate([yA, yE])

        # 3D-Var update
        self.theta, self.B = three_d_var_update(self.theta, self.B, self.H, y, self.R)

        # live plot update
        self.plotter.update(
            step=sim_time,
            theta_est=self.theta,
            sire_lat=self.sire_lat,
            sire_lon=self.sire_lon,
            measA_rad=yA,
            measE_rad=yE,
        )

        self.step_index += 1


    def close(self):
        self.plotter.close()
