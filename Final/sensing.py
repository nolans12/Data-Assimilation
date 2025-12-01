"""Simplified sensing utilities for a single ballistic target observed by GEO sats."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from common import ecef_to_lla, lla_to_ecef, sphere_line_intersection
from bias_field import bias_model

EARTH_RADIUS_KM = 6378.0
GEO_ALTITUDE_KM = 36000.0

def measurement_model(target_pos: NDArray[np.float64], sat_pos: NDArray[np.float64]) -> tuple[float, float]:
    """Compute azimuth and elevation from satellite to target.
    
    Args:
        target_pos: Target position in ECEF [km]
        sat_pos: Satellite position in ECEF [km]
        
    Returns:
        (azimuth, elevation) in radians
    """
    rel = target_pos - sat_pos
    x, y, z = rel
    az = np.arctan2(y, x)
    rng_norm = np.linalg.norm(rel)
    elev = np.arcsin(z / rng_norm)
    return az, elev

@dataclass(slots=True)
class Measurement:
    """Container for az/el measurements."""
    time: float
    sat_position: NDArray[np.float64]
    alpha: float  # azimuth [rad]
    beta: float  # elevation [rad]
    R: NDArray[np.float64]  # 2x2 covariance [rad^2]
    
    def sub_out_est_bias(self, theta_est: NDArray[np.float64], est_pos: NDArray[np.float64]):
        """Adjust measurement by REMOVING bias using current Az/El and true range.
        
        Process:
        1. Use current Az/El measurement with estimated range to get ECEF position
        2. Convert that ECEF position to LLA
        3. Subtract estimated bias from LLA
        4. Convert corrected LLA back to ECEF
        5. Apply measurement model to get new Az/El
        
        Args:
            theta_est: Bias field parameter vector (used for bias_model)
            est_pos: [x, y, z] in ECEF [km] of the estimated target position
        """
        # Step 1: Compute range from satellite to estimate
        estimated_range = np.linalg.norm(est_pos - self.sat_position)
        
        # Step 2: Use current Az/El measurement with estimated range to get ECEF position
        # Convert Az/El to unit direction vector
        rel_x = np.cos(self.alpha) * np.cos(self.beta)
        rel_y = np.sin(self.alpha) * np.cos(self.beta)
        rel_z = np.sin(self.beta)
        direction = np.array([rel_x, rel_y, rel_z])
        
        # Project from satellite position along this direction at estimated range
        meas_ecef = self.sat_position + direction * estimated_range
        
        # Step 3: Convert measurement ECEF position to LLA
        lat, lon, alt = ecef_to_lla(meas_ecef[0], meas_ecef[1], meas_ecef[2])
        
        # Step 4: Get bias at this lat/lon and subtract it
        bA, bE = bias_model(lat, lon, theta_est)
        corrected_lat = lat - bA
        corrected_lon = lon - bE
        
        # Step 5: Convert corrected LLA back to ECEF
        corrected_ecef = lla_to_ecef(corrected_lat, corrected_lon, alt)
        
        # Step 6: Apply measurement model to get new Az/El
        new_az, new_el = measurement_model(corrected_ecef, self.sat_position)

        self.alpha = new_az
        self.beta = new_el


class GeoSatellite:
    """Fixed GEO satellite with a simple bearings-only sensor."""

    def __init__(self, name: str, position: NDArray[np.float64], noise_sigma: float):
        self.name = name
        self.position = position
        self._R = np.diag([noise_sigma**2, noise_sigma**2])

    def measure(
        self,
        time_s: float,
        target_pos: NDArray[np.float64],
        rng: np.random.Generator,
        USE_BIAS: bool = False,
    ) -> Measurement:
        # Do we want to apply the bias field to this measurement?
        if USE_BIAS:
            # Bias field is in Lat Lon. So convert target pos to Lat Lon, then apply bias field.
            lat, lon, alt = ecef_to_lla(target_pos[0], target_pos[1], target_pos[2])
            bA, bE = bias_model(lat, lon)
            # Now conert lat lon with bias back to ECEF coordinates
            target_pos = lla_to_ecef(lat + bA, lon + bE, alt)

        az, elev = measurement_model(target_pos, self.position)
        noise = rng.multivariate_normal(mean=[0.0, 0.0], cov=self._R)
        
        return Measurement(
            time=time_s,
            sat_position=self.position.copy(),
            alpha=az + noise[0],
            beta=elev + noise[1],
            R=self._R.copy(),
        )


def create_geo_satellites(positions: list[list[float]], noise_sigma: float) -> list['GeoSatellite']:
    """Create GEO satellites at specified lat/lon positions.
    
    Args:
        positions: List of [lat, lon] in degrees
        noise_sigma: Measurement noise sigma [rad]
    """
    satellites: list[GeoSatellite] = []
    radius = EARTH_RADIUS_KM + GEO_ALTITUDE_KM
    
    for idx, (lat, lon) in enumerate(positions):
        # Init Pos in ECEF from Lat Lon and GEO Altitude
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        pos = np.array([
            radius * np.cos(lat_rad) * np.cos(lon_rad),
            radius * np.cos(lat_rad) * np.sin(lon_rad),
            radius * np.sin(lat_rad),
        ])
        satellites.append(GeoSatellite(name=f"GEO-{idx+1}", position=pos, noise_sigma=noise_sigma))
    return satellites

