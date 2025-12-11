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

def h_lat(lat, lon, theta):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    x0,x1,x2,x3,x4,x5, *_ = theta
    return np.stack([
        x0*np.ones_like(phi),
        x1**3*np.sin(phi),
        x2**4*np.cos(phi)*np.cos(lam),
        x3**2*np.cos(phi)*np.sin(lam),
        x4**5*np.sin(2*phi)*np.cos(2*lam),
        x5*np.sin(2*phi)*np.sin(2*lam)
    ], axis=-1)

def h_lon(lat, lon, theta):
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    *_,x6,x7,x8,x9,x10,x11 = theta
    return np.stack([
        x6*np.ones_like(phi),
        x7**3*np.sin(phi),
        x8**3*np.cos(phi)*np.cos(lam),
        x9**2*np.cos(phi)*np.sin(lam),
        x10**5*np.sin(2*phi)*np.cos(2*lam),
        -x11*np.sin(0.5*phi)*np.sin(lam)
    ], axis=-1)

def jacobian_lat(lat_deg, lon_deg, theta):
    """Jacobian row block for h_phi wrt 12 parameters."""
    phi, lam = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    x0,x1,x2,x3,x4,x5, *_ = theta

    return np.stack([
        np.ones_like(phi),
        3*x1**2 * np.sin(phi),
        4*x2**3 * np.cos(phi)*np.cos(lam),
        2*x3     * np.cos(phi)*np.sin(lam),
        5*x4**4 * np.sin(2*phi)*np.cos(2*lam),
        np.sin(2*phi)*np.sin(2*lam)
    ], axis=-1)


def jacobian_lon(lat_deg, lon_deg, theta):
    """Jacobian row block for h_lambda wrt 12 parameters."""
    phi, lam = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    *_, x6,x7,x8,x9,x10,x11 = theta

    return np.stack([
        np.ones_like(phi),
        3*x7**2 * np.sin(phi),
        3*x8**2 * np.cos(phi)*np.cos(lam),
        2*x9     * np.cos(phi)*np.sin(lam),
        5*x10**4 * np.sin(2*phi)*np.cos(2*lam),
        -np.sin(0.5*phi)*np.sin(lam)
    ], axis=-1)



# ============================================================
# 3D-Var Update with Newtons Method
# ============================================================

def three_d_var_newtons_update(theta_b, B, y, R,
                               sire_lat, sire_lon):
    """
    Newton (Gauss–Newton) 3D-Var update for nonlinear bias model.
    Only theta is updated; B is unchanged.
    
    Required call structure:
        theta_new = three_d_var_newtons_update(theta, B, H, y, R)

    This function *automatically* recomputes:
        - nonlinear predicted h(theta)
        - block Jacobian H(theta)

    NOTE: Because H depends on theta and SIRE locations, we detect
          the lat/lon arrays from the closure variables if needed.
    """

    phi = sire_lat
    lam = sire_lon
    N = len(phi)

    # ------------------------------------------------------------
    # Evaluate nonlinear h(theta)
    # ------------------------------------------------------------
    h_lat_vals = h_lat(phi, lam, theta_b)    # N×6
    h_lon_vals = h_lon(phi, lam, theta_b)    # N×6

    h_pred = np.concatenate([
        np.sum(h_lat_vals, axis=1),
        np.sum(h_lon_vals, axis=1)
    ])  # shape (2N,)

    innov = h_pred - y   # matches ∇J form

    # ------------------------------------------------------------
    # Build block-Jacobian
    # ------------------------------------------------------------
    J_lat = jacobian_lat(phi, lam, theta_b)  # N×6
    J_lon = jacobian_lon(phi, lam, theta_b)  # N×6

    H_jac = np.block([
        [J_lat,              np.zeros((N,6))],
        [np.zeros((N,6)),    J_lon]
    ])  # (2N × 12)

    # ------------------------------------------------------------
    # Newton / Gauss–Newton update
    # ------------------------------------------------------------
    B_inv = np.linalg.inv(B)
    R_inv = np.linalg.inv(R)

    grad = H_jac.T @ R_inv @ innov + B_inv @ (theta_b - theta_b)  # background fixed → term = 0
    Hess = H_jac.T @ R_inv @ H_jac + B_inv

    delta = np.linalg.solve(Hess, grad)
    theta_new = theta_b - delta

    return theta_new

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

        # SIRE locations (true)
        self.sire_lat, self.sire_lon = fibonacci_sites(NUM_SIRES, seed=SEED)

        # R to sample from
        self.R = np.diag(np.full(2*self.N, self.sigma**2))

        # Initial estimate and cov
        self.theta = np.zeros(12)
        self.B = np.diag(np.full(12, np.deg2rad(0.4)**2))

        # History for parameter tracking
        self.theta_history: list[np.ndarray] = []
        self.time_history: list[float] = []

        # live plotter script
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
        B_Lat_true, B_Lon_true = bias_model(self.sire_lat, self.sire_lon)

        # add noise to the true bias measurements - this is meas vector
        y_Lat = np.deg2rad(B_Lat_true) + self.rng.normal(0, self.sigma, size=self.N)
        y_Lon  = np.deg2rad(B_Lon_true) + self.rng.normal(0, self.sigma, size=self.N)
        y = np.concatenate([y_Lat, y_Lon])

        # 3D-Var update
        self.theta = three_d_var_newtons_update(
            self.theta,
            self.B,
            y,
            self.R,
            self.sire_lat,
            self.sire_lon
        )

        # Store history for analysis/plots
        self.theta_history.append(self.theta.copy())
        self.time_history.append(sim_time)


        # live plot update
        self.plotter.update(
            step=sim_time,
            theta_est=self.theta,
            sire_lat=self.sire_lat,
            sire_lon=self.sire_lon,
            meas_lat_rad=y_Lat,
            meas_lon_rad=y_Lon,
        )

        self.step_index += 1


    def close(self):
        self.plotter.close()

        # -------------------------------------------------------
        # Plot tracking of each theta parameter vs time
        # -------------------------------------------------------
        if not self.theta_history:
            return

        times = np.array(self.time_history)
        theta_hist = np.vstack(self.theta_history)  # (T, 12)
        theta_true = np.asarray(TRUE_THETA)         # (12,)

        # Convert time to minutes for plotting
        time_min = times / 60.0

        fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
        axes = axes.ravel()

        for i in range(12):
            ax = axes[i]
            ax.plot(time_min, theta_hist[:, i], label="Estimate", color="tab:blue")
            ax.axhline(theta_true[i], color="k", linestyle="--", label="Truth")
            ax.set_title(f"$\\theta_{i}$", fontsize=10)
            ax.grid(True, alpha=0.3)
            if i % 4 == 0:
                ax.set_ylabel("Value [rad]")

        for ax in axes[-4:]:
            ax.set_xlabel("Time [min]")

        # Single legend and title
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle("Bias Parameter Tracking vs Truth", fontsize=14, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        plt.show()



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

        # ------------------------------------------------------
        # Live parameter tracking figure (12 parameters)
        # ------------------------------------------------------
        self.fig_theta, theta_axes = plt.subplots(3, 4, figsize=(18, 8), sharex=True)
        self.theta_axes = theta_axes.ravel()
        self.theta_lines = []
        self.theta_time_hist: list[float] = []
        self.theta_hist: list[np.ndarray] = []

        for i, ax in enumerate(self.theta_axes):
            # Line for estimate
            (line_est,) = ax.plot([], [], color="tab:blue", label="Estimate")
            # Truth as horizontal line
            ax.axhline(TRUE_THETA[i], color="k", linestyle="--", label="Truth")

            ax.set_title(f"$\\theta_{i}$", fontsize=10)
            ax.grid(True, alpha=0.3)
            if i % 4 == 0:
                ax.set_ylabel("Value [rad]")

            self.theta_lines.append(line_est)

        for ax in self.theta_axes[-4:]:
            ax.set_xlabel("Time [min]")

        handles, labels = self.theta_axes[0].get_legend_handles_labels()
        self.fig_theta.legend(handles, labels, loc="upper center", ncol=2)
        self.fig_theta.suptitle("Bias Parameter Tracking vs Truth (Live)", fontsize=14, weight="bold")
        self.fig_theta.tight_layout(rect=[0, 0, 1, 0.92])

    def update(self, step, theta_est, sire_lat, sire_lon, meas_lat_rad, meas_lon_rad):

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

        # ------------------------------------------------------
        # Update live theta tracking figure
        # ------------------------------------------------------
        time_min = step / 60.0
        self.theta_time_hist.append(time_min)
        self.theta_hist.append(theta_est.copy())

        times = np.array(self.theta_time_hist)
        theta_arr = np.vstack(self.theta_hist)  # (T, 12)

        for i, line in enumerate(self.theta_lines):
            line.set_data(times, theta_arr[:, i])
            ax = self.theta_axes[i]
            ax.relim()
            ax.autoscale_view()

        self.fig_theta.canvas.draw()
        self.fig_theta.canvas.flush_events()

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