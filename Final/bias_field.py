import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Field we are trying to estimate
TRUE_THETA = np.deg2rad([
    -0.1, 0.1, 0.2, -0.15, -0.05, 0.3,   # Lattiude Params
   0, -0.2, 0.3, -0.2, 0.25, 0.15        # Longitude Params
])

def bias_model(lat_deg, lon_deg, theta=TRUE_THETA):
    """Compute azimuth and elevation bias fields (degrees) from param vector."""
    phi, lam = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11 = theta

    b_lat = (
        x0 + x1**3*np.sin(phi)
        + x2**4*np.cos(phi)*np.cos(lam)
        + x3**2*np.cos(phi)*np.sin(lam)
        + x4**5*np.sin(2*phi)*np.cos(2*lam)
        + x5*np.sin(2*phi)*np.sin(2*lam)
    )

    b_lon = (
        x6 + x7**3*np.sin(phi)
        + x8**3*np.cos(phi)*np.cos(lam)
        + x9**2*np.cos(phi)*np.sin(lam)
        + x10**5*np.sin(2*phi)*np.cos(2*lam)
        - x11*np.sin(0.5*phi)*np.sin(lam)
    )

    return np.rad2deg(b_lat), np.rad2deg(b_lon)

lat = np.linspace(-90,90,181)
lon = np.linspace(-180,180,361)
Lon, Lat = np.meshgrid(lon, lat)
bLat, bLon = bias_model(Lat, Lon)


def fibonacci_sites(n, seed=0):
    """Generate n uniformly distributed points on a sphere using Fibonacci lattice."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    phi = np.arccos(1 - 2*(idx + 0.5)/n) - np.pi/2
    lam = (2*np.pi/(np.sqrt(5)+1)) * idx + rng.uniform(-0.1,0.1,size=n)
    return np.rad2deg(phi), ((np.rad2deg(lam)+180)%360 - 180)


def plot_sire_locations(n_sires=25, seed=0):
    """Plot SIRE site locations on a global map."""
    sire_lat, sire_lon = fibonacci_sites(n_sires, seed=seed)
    
    fig = plt.figure(figsize=(11, 5))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="#f4f2ec")
    ax.add_feature(cfeature.OCEAN, facecolor="#c6dbef")
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Plot SIRE locations
    ax.scatter(
        sire_lon, sire_lat,
        s=80, c='red', marker='^', edgecolors='black', linewidths=0.5,
        transform=ccrs.PlateCarree(), zorder=5, label=f"SIRE Sites (n={n_sires})"
    )
    
    ax.set_title(f"SIRE Ground Station Locations", fontsize=13, weight="bold")
    ax.legend(loc="lower left", fontsize=10)
    plt.show()


def plot_field(theta=TRUE_THETA, title="True Bias Field (deg)", cmap="coolwarm"):
    bLat, bLon = bias_model(Lat, Lon, theta)

    # Plot Azimuth
    fig_lat = plt.figure(figsize=(11,5))
    ax_lat = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax_lat.set_global()
    ax_lat.coastlines()
    ax_lat.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax_lat.add_feature(cfeature.LAND, facecolor="#f4f2ec")
    im_lat = ax_lat.pcolormesh(lon, lat, bLat, cmap=cmap, shading="auto", transform=ccrs.PlateCarree())
    contours_lat = ax_lat.contour(lon, lat, bLat, levels=10, colors='black', linewidths=0.5, 
                                alpha=0.6, transform=ccrs.PlateCarree())
    ax_lat.clabel(contours_lat, inline=True, fontsize=8, fmt='%.2f')
    gl_lat = ax_lat.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im_lat, ax=ax_lat, orientation="horizontal", pad=0.04, label="Bias (deg)", shrink=0.8)
    ax_lat.set_title("Latitude " + title, fontsize=13, weight="bold")

    # Plot Elevation
    fig_lon = plt.figure(figsize=(11,5))
    ax_lon = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax_lon.set_global()
    ax_lon.coastlines()
    ax_lon.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax_lon.add_feature(cfeature.LAND, facecolor="#f4f2ec")
    im_lon = ax_lon.pcolormesh(lon, lat, bLon, cmap=cmap, shading="auto", transform=ccrs.PlateCarree())
    contours_lon = ax_lon.contour(lon, lat, bLon, levels=10, colors='black', linewidths=0.5, 
                                alpha=0.6, transform=ccrs.PlateCarree())
    ax_lon.clabel(contours_lon, inline=True, fontsize=8, fmt='%.2f')
    gl_lon = ax_lon.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im_lon, ax=ax_lon, orientation="horizontal", pad=0.04, label="Bias (deg)", shrink=0.8)
    ax_lon.set_title("Longitude " + title, fontsize=13, weight="bold")
    plt.show()

# plot_field()
# plot_sire_locations()
