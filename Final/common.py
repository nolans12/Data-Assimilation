import numpy as np
from numpy import typing as npt

def ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    """
    Assumed given in km, so convert to meters

    Output is: lat (deg), lon (deg), alt (km)
    """
    x = x * 1000
    y = y * 1000
    z = z * 1000

    # WGS84 ellipsoid constants
    a = 6378137.0  # Semi-major axis (meters)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2  # Square of eccentricity

    # Computation
    b = a * (1 - f)  # Semi-minor axis
    ep2 = (a**2 - b**2) / b**2  # Second eccentricity squared
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lat = np.arctan2(z + ep2 * b * sin_theta**3, p - e2 * a * cos_theta**3)
    lon = np.arctan2(y, x)

    # Iterative calculation for altitude
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert latitude and longitude from radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt / 1000


def lla_to_ecef(lat: float, lon: float, alt: float) -> npt.NDArray:
    """Conversion of lat (deg), lon (deg), and alt (km) to ECEF coordinates in km"""
    # Define the WGS84 datum
    a = 6378137.0  # Semi-major axis (meters)
    b = 6356752.31424518  # Semi-minor axis (meters)
    e_squared = (a**2 - b**2) / a**2  # First eccentricity squared
    
    lat = np.radians(lat)
    lon = np.radians(lon)

    # Calculate the ECEF coordinates
    N = a / np.sqrt(1 - e_squared * np.sin(lat) ** 2)
    x = (N + alt * 1000) * np.cos(lat) * np.cos(lon)
    y = (N + alt * 1000) * np.cos(lat) * np.sin(lon)
    z = ((b**2 / a**2) * N + alt * 1000) * np.sin(lat)
    return np.array([x / 1000, y / 1000, z / 1000])