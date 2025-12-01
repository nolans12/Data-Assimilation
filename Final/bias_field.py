import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Field we are trying to estimate
TRUE_THETA = np.deg2rad([
    -0.2, 0.1, 0.2, -0.15, -0.05, 0.3,   # Azimuth params
   -0.03, 0.01, 0.03, 0.03, 0.01          # Elevation params
])

def bias_model(lat_deg, lon_deg, theta=TRUE_THETA):
    """Compute azimuth and elevation bias fields (degrees) from param vector."""
    phi, lam = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    a0,a1,a2,a3,a4,a5,e0,e1,e2,e3,e4 = theta

    b_alpha = (
        a0 + a1*np.sin(phi)
        + a2*np.cos(phi)*np.cos(lam)
        + a3*np.cos(phi)*np.sin(lam)
        + a4*np.sin(2*phi)*np.cos(2*lam)
        + a5*np.sin(2*phi)*np.sin(2*lam)
    )

    b_elev = (
        e0 + e1*0.5*(3*np.sin(phi)**2 - 1)
        + e2*np.cos(phi)*np.cos(lam)
        - e3*np.cos(phi)*np.sin(lam)
        + e4*np.sin(2*phi)
    )

    return np.rad2deg(b_alpha), np.rad2deg(b_elev)

lat = np.linspace(-90,90,181)
lon = np.linspace(-180,180,361)
Lon, Lat = np.meshgrid(lon, lat)
bA, bE = bias_model(Lat, Lon)

def plot_field(theta=TRUE_THETA, title="True Bias Field (deg)", cmap="coolwarm"):
    bA, bE = bias_model(Lat, Lon, theta)

    # Plot Azimuth
    fig_az = plt.figure(figsize=(11,5))
    ax_az = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax_az.set_global()
    ax_az.coastlines()
    ax_az.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax_az.add_feature(cfeature.LAND, facecolor="#f4f2ec")
    im_az = ax_az.pcolormesh(lon, lat, bA, cmap=cmap, shading="auto", transform=ccrs.PlateCarree())
    contours_az = ax_az.contour(lon, lat, bA, levels=10, colors='black', linewidths=0.5, 
                                alpha=0.6, transform=ccrs.PlateCarree())
    ax_az.clabel(contours_az, inline=True, fontsize=8, fmt='%.2f')
    gl_az = ax_az.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im_az, ax=ax_az, orientation="horizontal", pad=0.04, label="Bias (deg)", shrink=0.8)
    ax_az.set_title("Azimuth " + title, fontsize=13, weight="bold")

    # Plot Elevation
    fig_el = plt.figure(figsize=(11,5))
    ax_el = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax_el.set_global()
    ax_el.coastlines()
    ax_el.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax_el.add_feature(cfeature.LAND, facecolor="#f4f2ec")
    im_el = ax_el.pcolormesh(lon, lat, bE, cmap=cmap, shading="auto", transform=ccrs.PlateCarree())
    contours_el = ax_el.contour(lon, lat, bE, levels=10, colors='black', linewidths=0.5, 
                                alpha=0.6, transform=ccrs.PlateCarree())
    ax_el.clabel(contours_el, inline=True, fontsize=8, fmt='%.2f')
    gl_el = ax_el.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    plt.colorbar(im_el, ax=ax_el, orientation="horizontal", pad=0.04, label="Bias (deg)", shrink=0.8)
    ax_el.set_title("Elevation " + title, fontsize=13, weight="bold")
    plt.show()

# plot_field()
