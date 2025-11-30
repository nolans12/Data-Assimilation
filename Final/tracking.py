"""Centralized EKF for GEO sensing measurements."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from sensing import Measurement

def measurement_model(
    sat_pos: NDArray[np.float64], target_pos: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Measurement model: az/el from target pos to sat pos."""
    rel = target_pos - sat_pos
    x, y, z = rel
    az = np.arctan2(y, x)
    rng = np.linalg.norm(rel)
    elev = np.arcsin(z / rng)
    return np.array([az, elev])


def measurement_jacobian(
    sat_pos: NDArray[np.float64], target_pos: NDArray[np.float64]
) -> NDArray[np.float64]:
    rel = target_pos - sat_pos
    x, y, z = rel
    range_sq = np.dot(rel, rel)
    range_xy_sq = max(x**2 + y**2, 1e-9)
    range_xy = np.sqrt(range_xy_sq)

    # Derivatives for azimuth = atan2(y, x)
    d_az_dx = -y / range_xy_sq
    d_az_dy = x / range_xy_sq
    d_az_dz = 0.0

    # Derivatives for elevation = arcsin(z / range)
    denom = range_sq * range_xy
    d_el_dx = -(z * x) / denom
    d_el_dy = -(z * y) / denom
    d_el_dz = range_xy / range_sq

    H = np.zeros((2, 6))
    H[0, 0] = d_az_dx
    H[0, 1] = d_az_dy
    H[0, 2] = d_az_dz
    H[1, 0] = d_el_dx
    H[1, 1] = d_el_dy
    H[1, 2] = d_el_dz
    return H


@dataclass
class TrackerState:
    state: NDArray[np.float64]
    covariance: NDArray[np.float64]


class Tracker:
    """Constant-velocity EKF that fuses simultaneous measurements."""

    def __init__(
        self,
        initial_state: NDArray[np.float64],
        initial_covariance: NDArray[np.float64] | None = None,
        process_sigma: float = 0.01,
    ):
        self.state = initial_state.copy()
        # Initial covariance: 1000 m^2 for positions, 100 m^2/s^2 for velocities
        if initial_covariance is None:
            self.covariance = np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])  # [km^2, km^2, km^2, (km/s)^2, ...]
        else:
            self.covariance = initial_covariance.copy()
        self.process_sigma = process_sigma

    def step(self, dt: float, measurements: list[Measurement]) -> TrackerState:
        if dt > 0.0:
            self._predict(dt)
        if measurements:
            self._update(measurements)
        return TrackerState(self.state.copy(), self.covariance.copy())

    def _predict(self, dt: float) -> None:
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        Q = self.process_sigma**2 * np.array(
            [
                [dt**3 / 3, 0, 0, dt**2 / 2, 0, 0],
                [0, dt**3 / 3, 0, 0, dt**2 / 2, 0],
                [0, 0, dt**3 / 3, 0, 0, dt**2 / 2],
                [dt**2 / 2, 0, 0, dt, 0, 0],
                [0, dt**2 / 2, 0, 0, dt, 0],
                [0, 0, dt**2 / 2, 0, 0, dt],
            ]
        )
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

    def _update(self, measurements: list[Measurement]) -> None:
        z_blocks = []
        h_blocks = []
        H_rows = []
        R_blocks = []
        for meas in measurements:
            z_blocks.append([meas.alpha, meas.beta])
            h_blocks.append(measurement_model(meas.sat_position, self.state[:3]))
            H_rows.append(measurement_jacobian(meas.sat_position, self.state[:3]))
            R_blocks.append(meas.R)

        z = np.concatenate(z_blocks)
        h = np.concatenate(h_blocks)
        H = np.vstack(H_rows)
        R = block_diag(R_blocks)

        innovation = z - h
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        I = np.eye(6)
        self.covariance = (I - K @ H) @ self.covariance @ (I - K @ H).T + K @ R @ K.T


def block_diag(blocks: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    total_dim = sum(block.shape[0] for block in blocks)
    result = np.zeros((total_dim, total_dim))
    offset = 0
    for block in blocks:
        dim = block.shape[0]
        result[offset : offset + dim, offset : offset + dim] = block
        offset += dim
    return result


