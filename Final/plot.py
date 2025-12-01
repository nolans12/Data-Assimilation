"""Plotting file"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import chi2
from common import ecef_to_lla

EARTH_RADIUS_KM = 6378.0
DATA_FILE = Path(__file__).resolve().parent / "data" / "simulation_run.npz"


class LivePlotter:
    """Real-time ground track visualization during simulation."""

    def __init__(self, zoom_extent: list[float] | None = None, save_frames: bool = True):
        """
        Args:
            zoom_extent: [min_lat, min_lon, max_lat, max_lon] for plot extent
            save_frames: If True, save frames for creating animation
        """
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        
        self.ax_ground = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        if zoom_extent:
            self.ax_ground.set_extent(
                [zoom_extent[1], zoom_extent[3], zoom_extent[0], zoom_extent[2]],
                crs=ccrs.PlateCarree()
            )
        else:
            self.ax_ground.set_global()
            
        self.ax_ground.coastlines(linewidth=0.5)
        self.ax_ground.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        self.ax_ground.add_feature(cfeature.LAND, facecolor="#f4f2ec")
        self.ax_ground.add_feature(cfeature.OCEAN, facecolor="#c6dbef")
        gl = self.ax_ground.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )
        gl.top_labels = False
        gl.right_labels = False
        self.ax_ground.set_title("Live Ballistic Trajectory Ground Track", fontsize=14, weight="bold")

        # Storage for ground track lines
        self.truth_track = None
        self.est_track = None
        self.truth_marker = None
        self.est_marker = None

        # Storage for satellites on ground track
        self.sat_markers = []
        
        # Storage for measurement markers
        self.meas_scatter = None
        
        # Frame saving for animation
        self.save_frames = save_frames
        self.frame_dir = Path(__file__).resolve().parent / "data" / "frames"
        self.frame_count = 0
        if self.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            # Clean old frames
            for old_frame in self.frame_dir.glob("frame_*.png"):
                old_frame.unlink()

        plt.tight_layout()
        plt.pause(0.001)

    def update(
        self,
        time: float,
        truth_state: np.ndarray,
        est_state: np.ndarray,
        cov_diag: np.ndarray,
        history_times: list[float],
        history_truth: list[np.ndarray],
        history_est: list[np.ndarray],
        history_cov: list[np.ndarray],
        satellite_positions: list[np.ndarray] | None = None,
        measurements_lla: np.ndarray | None = None,
    ) -> None:
        """Update ground track with new data."""
        # Convert ECEF to LLA
        truth_lla = ecef_to_lla(truth_state[0], truth_state[1], truth_state[2])
        est_lla = ecef_to_lla(est_state[0], est_state[1], est_state[2])

        # Ground track update
        if len(history_truth) > 1:
            truth_llas = np.array(
                [ecef_to_lla(s[0], s[1], s[2]) for s in history_truth]
            )
            est_llas = np.array([ecef_to_lla(s[0], s[1], s[2]) for s in history_est])

            if self.truth_track:
                self.truth_track.remove()
            if self.est_track:
                self.est_track.remove()

            (self.truth_track,) = self.ax_ground.plot(
                truth_llas[:, 1],
                truth_llas[:, 0],
                "b-",
                linewidth=2.5,
                label="Truth",
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            (self.est_track,) = self.ax_ground.plot(
                est_llas[:, 1],
                est_llas[:, 0],
                "r--",
                linewidth=2,
                label="Estimate",
                transform=ccrs.PlateCarree(),
                zorder=2,
            )

        # Current position markers
        if self.truth_marker:
            self.truth_marker.remove()
        if self.est_marker:
            self.est_marker.remove()

        (self.truth_marker,) = self.ax_ground.plot(
            truth_lla[1],
            truth_lla[0],
            "bo",
            markersize=10,
            label="Current Truth",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        (self.est_marker,) = self.ax_ground.plot(
            est_lla[1],
            est_lla[0],
            "ro",
            markersize=10,
            label="Current Estimate",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        # Plot satellite positions
        if satellite_positions:
            for marker in self.sat_markers:
                marker.remove()
            self.sat_markers = []

            for sat_pos in satellite_positions:
                sat_lla = ecef_to_lla(sat_pos[0], sat_pos[1], sat_pos[2])
                (marker,) = self.ax_ground.plot(
                    sat_lla[1],
                    sat_lla[0],
                    "g^",
                    markersize=12,
                    label="GEO Satellites" if not self.sat_markers else "",
                    transform=ccrs.PlateCarree(),
                    zorder=6,
                )
                self.sat_markers.append(marker)
        
        # Plot measurements as red X's
        if measurements_lla is not None and len(measurements_lla) > 0:
            if self.meas_scatter:
                self.meas_scatter.remove()
            self.meas_scatter = self.ax_ground.scatter(
                measurements_lla[:, 1],
                measurements_lla[:, 0],
                marker="x",
                s=30,
                c="red",
                alpha=0.5,
                label="Measurements",
                transform=ccrs.PlateCarree(),
                zorder=7,
            )

        # Update legend
        if len(history_truth) == 1:
            self.ax_ground.legend(loc="best", fontsize=10)
        
        # Add time info to title
        time_min = time / 60.0
        self.ax_ground.set_title(
            f"Live Ballistic Trajectory Ground Track (t={time_min:.2f} min)", 
            fontsize=14, 
            weight="bold"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        
        # Save frame for animation
        if self.save_frames:
            frame_path = self.frame_dir / f"frame_{self.frame_count:04d}.png"
            self.fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            self.frame_count += 1

    def close(self) -> None:
        plt.ioff()
        
        # Create animation from saved frames
        if self.save_frames and self.frame_count > 0:
            print(f"\nCreating animation from {self.frame_count} frames...")
            output_dir = self.frame_dir.parent
            
            try:
                import imageio
                # Create GIF
                gif_path = output_dir / "live_plot_animation.gif"
                frames = []
                for i in range(self.frame_count):
                    frame_path = self.frame_dir / f"frame_{i:04d}.png"
                    if frame_path.exists():
                        frames.append(imageio.imread(frame_path))
                
                if frames:
                    imageio.mimsave(gif_path, frames, fps=10, loop=0)
                    print(f"Saved animation GIF to {gif_path}")
                    
                    # Try to create MP4 if ffmpeg is available
                    try:
                        mp4_path = output_dir / "live_plot_animation.mp4"
                        imageio.mimsave(mp4_path, frames, fps=10, codec='libx264')
                        print(f"Saved animation MP4 to {mp4_path}")
                    except Exception as e:
                        print(f"Could not create MP4 (ffmpeg may not be installed): {e}")
                
            except ImportError:
                print("imageio not installed. Install with 'pip install imageio' to create animations.")
            except Exception as e:
                print(f"Error creating animation: {e}")
        
        plt.show()


def plot_final_results(data_path: Path = DATA_FILE, zoom_extent: list[float] | None = None) -> None:
    """Create 3x1 subplot: altitude profile, velocity magnitude profile, and ground track.
    
    Args:
        data_path: Path to simulation data
        zoom_extent: [min_lat, min_lon, max_lat, max_lon] for ground track zoom
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"No simulation results found at {data_path}. Run main.py first."
        )

    data = np.load(data_path)
    times = data["times"]
    truth = data["truth"]
    estimates = data["estimates"]
    cov_diagonals = data["cov_diagonals"]
    
    # Load measurements if available
    meas_sat_positions = data.get("meas_sat_positions", np.array([]))
    meas_alphas = data.get("meas_alphas", np.array([]))
    meas_betas = data.get("meas_betas", np.array([]))
    meas_truth_positions = data.get("meas_truth_positions", np.array([]))

    # Convert to LLA
    truth_lla = np.array([ecef_to_lla(s[0], s[1], s[2]) for s in truth])
    est_lla = np.array([ecef_to_lla(s[0], s[1], s[2]) for s in estimates])
    
    # Convert measurements to LLA using az/el + true range
    meas_lla = None
    if len(meas_truth_positions) > 0 and len(meas_sat_positions) > 0:
        meas_target_positions = []
        for sat_pos, alpha, beta, true_pos in zip(meas_sat_positions, meas_alphas, meas_betas, meas_truth_positions):
            true_range = np.linalg.norm(true_pos - sat_pos)
            rel_x = np.cos(alpha) * np.cos(beta)
            rel_y = np.sin(alpha) * np.cos(beta)
            rel_z = np.sin(beta)
            meas_pos = sat_pos + np.array([rel_x, rel_y, rel_z]) * true_range
            meas_target_positions.append(meas_pos)
        meas_lla = np.array([ecef_to_lla(p[0], p[1], p[2]) for p in meas_target_positions])

    # Compute altitude and velocity magnitudes
    truth_alt = truth_lla[:, 2]  # Altitude in km
    est_alt = est_lla[:, 2]
    
    truth_vel_mag = np.linalg.norm(truth[:, 3:6], axis=1)
    est_vel_mag = np.linalg.norm(estimates[:, 3:6], axis=1)
    
    # Compute velocity magnitude uncertainty (propagate covariance)
    # For v = [vx, vy, vz], |v| = sqrt(vx^2 + vy^2 + vz^2)
    # Approximate uncertainty: σ_|v| ≈ sqrt((vx*σ_vx)^2 + (vy*σ_vy)^2 + (vz*σ_vz)^2) / |v|
    vel_est = estimates[:, 3:6]
    vel_std = np.sqrt(cov_diagonals[:, 3:6])
    vel_mag_unc = np.sqrt(
        (vel_est[:, 0] * vel_std[:, 0])**2 + 
        (vel_est[:, 1] * vel_std[:, 1])**2 + 
        (vel_est[:, 2] * vel_std[:, 2])**2
    ) / (est_vel_mag + 1e-10)  # Avoid division by zero
    
    # Altitude uncertainty (Z-component std dev)
    alt_std = np.sqrt(cov_diagonals[:, 2])

    # Create 3x1 subplot figure
    plt.figure(figsize=(16, 12))
    
    # 1. Altitude Profile with ±1σ uncertainty band
    ax_alt = plt.subplot(3, 1, 1)
    ax_alt.plot(times, truth_alt, "b-", linewidth=2.5, label="Truth", alpha=0.4, zorder=2)
    ax_alt.plot(times, est_alt, "r--", linewidth=2, label="Estimate", zorder=3)
    
    # ±1σ uncertainty band
    ax_alt.fill_between(
        times, 
        est_alt - alt_std, 
        est_alt + alt_std, 
        color="red", 
        alpha=0.2, 
        label="±1σ uncertainty",
        zorder=1
    )
    
    ax_alt.set_title("Altitude Profile", fontsize=14, weight="bold")
    ax_alt.set_xlabel("Time [s]", fontsize=12)
    ax_alt.set_ylabel("Altitude [km]", fontsize=12)
    ax_alt.legend(loc="best", fontsize=11)
    ax_alt.grid(True, alpha=0.5, linestyle='--', linewidth=0.7)
    
    # 2. Velocity Magnitude Profile with ±1σ uncertainty band
    ax_vel = plt.subplot(3, 1, 2)
    ax_vel.plot(times, truth_vel_mag, "b-", linewidth=2.5, label="Truth", alpha=0.4, zorder=2)
    ax_vel.plot(times, est_vel_mag, "r--", linewidth=2, label="Estimate", zorder=3)
    
    # ±1σ uncertainty band
    ax_vel.fill_between(
        times, 
        est_vel_mag - vel_mag_unc, 
        est_vel_mag + vel_mag_unc, 
        color="red", 
        alpha=0.2, 
        label="±1σ uncertainty",
        zorder=1
    )
    
    ax_vel.set_title("Velocity Magnitude Profile", fontsize=14, weight="bold")
    ax_vel.set_xlabel("Time [s]", fontsize=12)
    ax_vel.set_ylabel("Velocity [km/s]", fontsize=12)
    ax_vel.legend(loc="best", fontsize=11)
    ax_vel.grid(True, alpha=0.5, linestyle='--', linewidth=0.7)
    
    # 3. Ground Track
    ax_ground = plt.subplot(3, 1, 3, projection=ccrs.PlateCarree())
    
    if zoom_extent:
        ax_ground.set_extent(
            [zoom_extent[1], zoom_extent[3], zoom_extent[0], zoom_extent[2]],
            crs=ccrs.PlateCarree()
        )
    else:
        ax_ground.set_global()
        
    ax_ground.coastlines(linewidth=0.5)
    ax_ground.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax_ground.add_feature(cfeature.LAND, facecolor="#f4f2ec")
    ax_ground.add_feature(cfeature.OCEAN, facecolor="#c6dbef")
    gl = ax_ground.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    ax_ground.plot(
        truth_lla[:, 1],
        truth_lla[:, 0],
        "b-",
        linewidth=2.5,
        label="Truth",
        transform=ccrs.PlateCarree(),
        zorder=3
    )
    ax_ground.plot(
        est_lla[:, 1],
        est_lla[:, 0],
        "r--",
        linewidth=2,
        label="Estimate",
        transform=ccrs.PlateCarree(),
        zorder=2
    )
    ax_ground.plot(
        truth_lla[0, 1],
        truth_lla[0, 0],
        "go",
        markersize=10,
        label="Launch",
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    ax_ground.plot(
        truth_lla[-1, 1],
        truth_lla[-1, 0],
        "ko",
        markersize=10,
        label="Impact",
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    
    # Plot measurements as red X's
    if meas_lla is not None and len(meas_lla) > 0:
        ax_ground.scatter(
            meas_lla[:, 1],
            meas_lla[:, 0],
            marker="x",
            s=50,
            c="red",
            alpha=0.6,
            label="Measurements",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
    
    ax_ground.set_title("Ballistic Trajectory Ground Track", fontsize=14, weight="bold")
    ax_ground.legend(loc="best", fontsize=11)

    plt.tight_layout()

    # Save figure (PNG and PDF)
    output_dir = data_path.parent
    output_path_png = output_dir / "tracking_results.png"
    output_path_pdf = output_dir / "tracking_results.pdf"
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved tracking results to {output_path_png} and {output_path_pdf}")

    plt.show(block=False)


def plot_error_analysis(data_path: Path = DATA_FILE) -> None:
    """Create 3x3 subplot showing component-wise errors with ±1σ bounds and NEES.
    
    Args:
        data_path: Path to simulation data
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"No simulation results found at {data_path}. Run main.py first."
        )

    data = np.load(data_path)
    times = data["times"]
    truth = data["truth"]
    estimates = data["estimates"]
    cov_diagonals = data["cov_diagonals"]
    
    # Compute errors for each component
    errors = truth - estimates
    pos_errors = errors[:, :3] * 1000  # Convert km to meters
    vel_errors = errors[:, 3:6] * 1000  # Convert km/s to m/s
    
    # Extract standard deviations
    pos_std = np.sqrt(cov_diagonals[:, :3]) * 1000  # Convert km to meters
    vel_std = np.sqrt(cov_diagonals[:, 3:6]) * 1000  # Convert km/s to m/s
    
    # Compute NEES (Normalized Estimation Error Squared)
    n_steps = len(times)
    nees = np.zeros(n_steps)
    for i in range(n_steps):
        error = truth[i] - estimates[i]
        # Use diagonal covariance for inversion
        P_inv_diag = 1.0 / (cov_diagonals[i] + 1e-10)
        nees[i] = np.sqrt(np.sum(error**2 * P_inv_diag))
    
    # Chi-squared bounds for NEES (6 DOF for position + velocity)
    dof_nees = 6
    nees_lower = np.sqrt(chi2.ppf(0.025, dof_nees))
    nees_upper = np.sqrt(chi2.ppf(0.975, dof_nees))
    
    # Compute consistency percentage
    nees_consistent = np.sum((nees >= nees_lower) & (nees <= nees_upper)) / len(nees) * 100
    
    # Create 3x3 subplot figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    fig.suptitle("Component-Wise Estimation Errors with ±1σ Bounds", fontsize=16, weight="bold")
    
    # Position error plots (top row)
    pos_labels = ["X Position Error", "Y Position Error", "Z Position Error"]
    pos_components = ["x", "y", "z"]
    
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        
        # Plot error
        ax.plot(times, pos_errors[:, i], "k-", linewidth=2, label="Error", zorder=3)
        
        # Plot ±1σ bounds
        ax.plot(times, pos_std[:, i], "r--", linewidth=1.5, label="+1σ", zorder=2)
        ax.plot(times, -pos_std[:, i], "r--", linewidth=1.5, label="-1σ", zorder=2)
        
        # Fill ±1σ region
        ax.fill_between(
            times,
            -pos_std[:, i],
            pos_std[:, i],
            color="red",
            alpha=0.15,
            label="±1σ region",
            zorder=1
        )
        
        # Plot zero line
        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=0)
        
        ax.set_title(pos_labels[i], fontsize=12, weight="bold")
        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel(f"Error in {pos_components[i]} [m]", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc="best", fontsize=9)
    
    # Velocity error plots (middle row)
    vel_labels = ["X Velocity Error", "Y Velocity Error", "Z Velocity Error"]
    vel_components = ["vx", "vy", "vz"]
    
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        
        # Plot error
        ax.plot(times, vel_errors[:, i], "k-", linewidth=2, label="Error", zorder=3)
        
        # Plot ±1σ bounds
        ax.plot(times, vel_std[:, i], "r--", linewidth=1.5, label="+1σ", zorder=2)
        ax.plot(times, -vel_std[:, i], "r--", linewidth=1.5, label="-1σ", zorder=2)
        
        # Fill ±1σ region
        ax.fill_between(
            times,
            -vel_std[:, i],
            vel_std[:, i],
            color="red",
            alpha=0.15,
            label="±1σ region",
            zorder=1
        )
        
        # Plot zero line
        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=0)
        
        ax.set_title(vel_labels[i], fontsize=12, weight="bold")
        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel(f"Error in {vel_components[i]} [m/s]", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc="best", fontsize=9)
    
    # NEES plot (bottom row, spanning all 3 columns)
    ax_nees = fig.add_subplot(gs[2, :])
    ax_nees.plot(times, nees, "b-", linewidth=2, label="NEES", zorder=3)
    ax_nees.axhline(nees_lower, color="r", linestyle="--", linewidth=2, label=f"95% bounds (χ²({dof_nees}))", zorder=2)
    ax_nees.axhline(nees_upper, color="r", linestyle="--", linewidth=2, zorder=2)
    ax_nees.fill_between(times, nees_lower, nees_upper, color="red", alpha=0.1, zorder=1)
    ax_nees.axhline(np.sqrt(dof_nees), color="g", linestyle=":", linewidth=1.5, label="Expected value", zorder=2)
    ax_nees.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=0)
    
    ax_nees.set_title(f"NEES (Normalized Estimation Error Squared) - Consistency: {nees_consistent:.1f}%", 
                      fontsize=12, weight="bold")
    ax_nees.set_xlabel("Time [s]", fontsize=11)
    ax_nees.set_ylabel("NEES", fontsize=11)
    ax_nees.legend(loc="best", fontsize=9)
    ax_nees.grid(True, alpha=0.3)
    ax_nees.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save figure (PNG and PDF)
    output_dir = data_path.parent
    output_path_png = output_dir / "error_analysis.png"
    output_path_pdf = output_dir / "error_analysis.pdf"
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved error analysis to {output_path_png} and {output_path_pdf}")
    
    plt.show(block=False)


def plot_measurement_errors(data_path: Path = DATA_FILE) -> None:
    """Create 3x1 subplot showing measurement errors: LOS error, azimuth error, elevation error.
    
    Args:
        data_path: Path to simulation data
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"No simulation results found at {data_path}. Run main.py first."
        )

    data = np.load(data_path)
    times = data["times"]
    truth = data["truth"]
    
    # Load measurements
    meas_times = data.get("meas_times", np.array([]))
    meas_sat_positions = data.get("meas_sat_positions", np.array([]))
    meas_alphas = data.get("meas_alphas", np.array([]))
    meas_betas = data.get("meas_betas", np.array([]))
    
    if len(meas_times) == 0:
        print("No measurements available for error analysis")
        return
    
    # Compute expected measurements from truth
    # For each measurement, find the corresponding truth position
    los_errors = []
    az_errors = []
    el_errors = []
    meas_times_list = []
    
    for i, (meas_time, sat_pos, meas_alpha, meas_beta) in enumerate(
        zip(meas_times, meas_sat_positions, meas_alphas, meas_betas)
    ):
        # Find closest truth time
        time_idx = np.argmin(np.abs(times - meas_time))
        true_pos = truth[time_idx, :3]
        
        # Compute expected measurement (true az/el from true position)
        rel = true_pos - sat_pos
        rel_norm = np.linalg.norm(rel)
        
        # Expected azimuth and elevation
        true_alpha = np.arctan2(rel[1], rel[0])
        true_beta = np.arcsin(rel[2] / rel_norm) if rel_norm > 0 else 0
        
        # Compute errors
        # Azimuth error (wrap to [-pi, pi])
        az_error = meas_alpha - true_alpha
        az_error = np.arctan2(np.sin(az_error), np.cos(az_error))  # Wrap to [-pi, pi]
        
        # Elevation error
        el_error = meas_beta - true_beta
        
        # LOS error (absolute angular error)
        # Compute angle between measured and true LOS vectors
        # Measured LOS direction
        meas_los = np.array([
            np.cos(meas_alpha) * np.cos(meas_beta),
            np.sin(meas_alpha) * np.cos(meas_beta),
            np.sin(meas_beta)
        ])
        # True LOS direction
        true_los = rel / rel_norm
        
        # Angular separation (absolute LOS error)
        cos_angle = np.clip(np.dot(meas_los, true_los), -1.0, 1.0)
        los_error = np.arccos(cos_angle)
        
        los_errors.append(los_error)
        az_errors.append(az_error)
        el_errors.append(el_error)
        meas_times_list.append(meas_time)
    
    los_errors = np.array(los_errors)
    az_errors = np.array(az_errors)
    el_errors = np.array(el_errors)
    meas_times_min = np.array(meas_times_list) / 60.0  # Convert to minutes
    
    # Create 3x1 subplot figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Measurement Errors vs Expected (True) Measurements", fontsize=16, weight="bold")
    
    # 1. LOS Error (absolute angular error)
    ax_los = axes[0]
    ax_los.plot(meas_times_min, los_errors * 1e6, "ko", markersize=3, alpha=0.6)  # Convert to microradians
    ax_los.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax_los.set_title("Line-of-Sight (LOS) Angular Error", fontsize=13, weight="bold")
    ax_los.set_xlabel("Time [min]", fontsize=11)
    ax_los.set_ylabel("LOS Error [µrad]", fontsize=11)
    ax_los.grid(True, alpha=0.3)
    ax_los.set_ylim(bottom=0)
    
    # Add statistics
    los_mean = np.mean(los_errors) * 1e6
    los_std = np.std(los_errors) * 1e6
    ax_los.text(0.98, 0.98, f"Mean: {los_mean:.1f} µrad\nStd: {los_std:.1f} µrad",
                transform=ax_los.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    # 2. Azimuth Error (signed)
    ax_az = axes[1]
    ax_az.plot(meas_times_min, az_errors * 1e6, "bo", markersize=3, alpha=0.6)  # Convert to microradians
    ax_az.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax_az.set_title("Azimuth Error (Measured - True)", fontsize=13, weight="bold")
    ax_az.set_xlabel("Time [min]", fontsize=11)
    ax_az.set_ylabel("Azimuth Error [µrad]", fontsize=11)
    ax_az.grid(True, alpha=0.3)
    
    # Add statistics
    az_mean = np.mean(az_errors) * 1e6
    az_std = np.std(az_errors) * 1e6
    ax_az.text(0.98, 0.98, f"Mean: {az_mean:.1f} µrad\nStd: {az_std:.1f} µrad",
               transform=ax_az.transAxes, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
               fontsize=10)
    
    # 3. Elevation Error (signed)
    ax_el = axes[2]
    ax_el.plot(meas_times_min, el_errors * 1e6, "ro", markersize=3, alpha=0.6)  # Convert to microradians
    ax_el.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax_el.set_title("Elevation Error (Measured - True)", fontsize=13, weight="bold")
    ax_el.set_xlabel("Time [min]", fontsize=11)
    ax_el.set_ylabel("Elevation Error [µrad]", fontsize=11)
    ax_el.grid(True, alpha=0.3)
    
    # Add statistics
    el_mean = np.mean(el_errors) * 1e6
    el_std = np.std(el_errors) * 1e6
    ax_el.text(0.98, 0.98, f"Mean: {el_mean:.1f} µrad\nStd: {el_std:.1f} µrad",
               transform=ax_el.transAxes, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
               fontsize=10)
    
    plt.tight_layout()
    
    # Save figure (PNG and PDF)
    output_dir = data_path.parent
    output_path_png = output_dir / "measurement_errors.png"
    output_path_pdf = output_dir / "measurement_errors.pdf"
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved measurement error analysis to {output_path_png} and {output_path_pdf}")
    
    plt.show(block=False)


if __name__ == "__main__":
    from main import ZOOM
    plot_final_results(zoom_extent=ZOOM)
    plot_error_analysis()
    plot_measurement_errors()
    plt.show()  # Keep all plots open