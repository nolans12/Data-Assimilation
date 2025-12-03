"""ASEN 6055 Final Project for Ballistic Tracking"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from sensing import create_geo_satellites
from target import BallisticTarget
from bias_field import bias_model, TRUE_THETA
from ekf_tracker import EKF
from bias_estimator import BiasFieldEstimatorLive
from plot import LivePlotter
from common import ecef_to_lla

# RNG Seed
SEED = 0
rng = np.random.default_rng(SEED)

# Whether to adjust measurements by bias field or not
USE_BIAS = True
# Whether to estimate bias field parameters and correct measurements by them or not
ESTIMATE_BIAS = True
# Number of SIRE sites for bias esimtation
NUM_SIRES = 25
# Noise of estimating bias locations
BIAS_NOISE_SIGMA = np.deg2rad(0.1) # for visual, easy to see

# Meas Noise (sigma)
NOISE_SIGMA = 10e-6 # radians

# Process Noise (sigma)
PROCESS_SIGMA = 100 / 1000 # km/s for constant velocity model

# Sim Parameters
SIM_DURATION_S = 40 * 60 # sec
DT_S = 10.0 # how often to take a meas and update ekf
LAUNCH = [-7.0, -42.0, 0.0] # lat, lon, alt
IMPACT = [54.0, 160.0, 0.0] # lat, lon, alt
MAX_ALTITUDE_KM = 250.0 # km, max alt of trajectory
ZOOM_MAG = 15.0
ZOOM = [LAUNCH[0] - ZOOM_MAG, LAUNCH[1] - ZOOM_MAG, IMPACT[0] + ZOOM_MAG, IMPACT[1] + ZOOM_MAG]

DATA_DIR = Path(__file__).resolve().parent / "data"

def run_simulation(seed: int = 0, live_plot: bool = True) -> Path:
    print(f"LAUNCH: lat={LAUNCH[0]}°, lon={LAUNCH[1]}°, alt={LAUNCH[2]} km")
    print(f"IMPACT: lat={IMPACT[0]}°, lon={IMPACT[1]}°, alt={IMPACT[2]} km")
    print(f"MAX ALTITUDE: {MAX_ALTITUDE_KM} km")
    
    #### CREATE TARGET ####
    target = BallisticTarget(
        launch_lat_deg=LAUNCH[0],
        launch_lon_deg=LAUNCH[1],
        launch_alt_km=LAUNCH[2],
        impact_lat_deg=IMPACT[0],
        impact_lon_deg=IMPACT[1],
        impact_alt_km=IMPACT[2],
        flight_duration_s=SIM_DURATION_S,
        max_altitude_km=MAX_ALTITUDE_KM,
    )
    
    #### CREATE SATELLITES ####
    # One near launch, one near impact, one in between
    # w/ random +- 10 degree jitter
    jittered_positions = []
    for lat, lon in [
        LAUNCH[:2], 
        IMPACT[:2],  
        [(LAUNCH[0] + IMPACT[0])/2, (LAUNCH[1] + IMPACT[1])/2],  
    ]:
        jitter_lat = lat + rng.uniform(-10, 10)
        jitter_lon = lon + rng.uniform(-10, 10)
        jittered_positions.append([jitter_lat, jitter_lon])
    satellites = create_geo_satellites(
        positions=jittered_positions,
        noise_sigma=NOISE_SIGMA,    
    )

    #### CREATE BIAS ESTIMATOR ####    
    if ESTIMATE_BIAS:
        bias_estimator = BiasFieldEstimatorLive(
            NUM_SIRES=NUM_SIRES,
            NOISE_SIGMA=BIAS_NOISE_SIGMA,
            SEED=SEED,
            SAVE_FRAMES=True,
        )
    else:
        bias_estimator = None

    #### CREATE EKF ####
    state0, vel0 = target.state_at(0.0)
    init_state = np.concatenate([state0 + rng.normal(0, 5.0, size=3), vel0])
    ekf = EKF(initial_state=init_state, process_sigma=PROCESS_SIGMA)

    #### RUN SIMULATION ####
    truth = []
    estimates = []
    cov_diagonals = []
    all_measurements = []

    #### PLOT SIMULATION ####
    plotter = LivePlotter(zoom_extent=ZOOM) if live_plot else None

    # Times to take measurements at, every dt
    times = np.arange(0.0, SIM_DURATION_S + DT_S, DT_S)
    prev_time = times[0]
    for idx, t in enumerate(times):
        pos, vel = target.state_at(t)
        truth_state = np.concatenate([pos, vel])
        truth.append(truth_state)

        # Sample meas w/ R
        dt = t - prev_time
        measurements = [sat.measure(t, pos, rng=rng, USE_BIAS=USE_BIAS) for sat in satellites]
        all_measurements.extend(measurements)  
        
         # Bias Field Estimation
        if ESTIMATE_BIAS:
            bias_estimator.step(sim_time=t)
            # Adjust measurements based on bias_estimator!
            for meas in measurements:
                meas.sub_out_est_bias(theta_est=bias_estimator.theta, est_pos=pos)

        # Run EKF on measuremetns at time step
        ekf_state = ekf.step(dt, measurements)
        estimates.append(ekf_state.state)
        cov_diagonals.append(np.diag(ekf_state.covariance))

        # Update live plot every 5 steps
        if plotter:
            sat_positions = [sat.position for sat in satellites]
            
            # Convert measurements to LLA using az/el + distance from satellite to true target position (only use true pos for vis)
            meas_lla_list = []
            for idx, meas in enumerate(all_measurements):
                sat_pos = meas.sat_position
                alpha, beta = meas.alpha, meas.beta

                time_idx = idx // len(satellites)
                target_ecef = truth[time_idx][:3]
                true_range = np.linalg.norm(target_ecef - sat_pos)

                # Az, El, R to ECEF
                rel_x = np.cos(alpha) * np.cos(beta)
                rel_y = np.sin(alpha) * np.cos(beta)
                rel_z = np.sin(beta)
                meas_pos = sat_pos + np.array([rel_x, rel_y, rel_z]) * true_range
                meas_lla_list.append(ecef_to_lla(meas_pos[0], meas_pos[1], meas_pos[2]))
            meas_lla = np.array(meas_lla_list) 

            # live plot
            plotter.update(
                time=t,
                truth_state=truth_state,
                est_state=ekf_state.state,
                cov_diag=np.diag(ekf_state.covariance),
                history_times=times[:idx+1].tolist(),
                history_truth=truth,
                history_est=estimates,
                history_cov=cov_diagonals,
                satellite_positions=sat_positions,
                measurements_lla=meas_lla,
            )

        prev_time = t

    if plotter:
        print("\nSimulation complete.")
        plotter.close()
        
    if ESTIMATE_BIAS:
        bias_estimator.close()


    #### SAVE RESULTS ####
    truth_arr = np.stack(truth)
    est_arr = np.stack(estimates)
    cov_arr = np.stack(cov_diagonals)
    
    #### SAVE MEASUREMENTS ####
    meas_times = np.array([m.time for m in all_measurements])
    meas_positions = np.array([m.sat_position for m in all_measurements])
    meas_alphas = np.array([m.alpha for m in all_measurements])
    meas_betas = np.array([m.beta for m in all_measurements])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "simulation_run.npz"
    np.savez(
        output_path,
        times=times,
        truth=truth_arr,
        estimates=est_arr,
        cov_diagonals=cov_arr,
        meas_times=meas_times,
        meas_sat_positions=meas_positions,
        meas_alphas=meas_alphas,
        meas_betas=meas_betas,
    )
    return output_path


if __name__ == "__main__":
    result_path = run_simulation()
    print(f"Saved ballistic tracking results to {result_path}")

