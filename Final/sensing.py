"""Simplified sensing utilities for a single ballistic target observed by GEO sats."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

EARTH_RADIUS_KM = 6378.0
GEO_ALTITUDE_KM = 36000.0

@dataclass(slots=True)
class Measurement:
    """Container for az/el measurements."""
    time: float
    sat_position: NDArray[np.float64]
    alpha: float  # azimuth [rad]
    beta: float  # elevation [rad]
    R: NDArray[np.float64]  # 2x2 covariance [rad^2]


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
    ) -> Measurement:
        rel = target_pos - self.position
        x, y, z = rel
        az = np.arctan2(y, x)
        rng_norm = np.linalg.norm(rel)
        elev = np.arcsin(z / rng_norm)
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

