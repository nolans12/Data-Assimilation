import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def bias_model(lat_deg, lon_deg, theta):
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

# Example parameter set (radians)
theta_true = np.deg2rad([
    0.1,  0.1, -0.2, 0.15, 0.05, -0.3,   # Azimuth params
   -0.05, 0.25, 0.1, 0.75, 0.3          # Elevation params
])

lat = np.linspace(-90,90,181)
lon = np.linspace(-180,180,361)
Lon, Lat = np.meshgrid(lon, lat)
bA, bE = bias_model(Lat, Lon, theta_true)

def plot_field(field, title, cmap="coolwarm"):
    fig = plt.figure(figsize=(11,5))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="#f4f2ec")
    im = ax.pcolormesh(lon, lat, field, cmap=cmap, shading="auto", transform=ccrs.PlateCarree())
    
    # Add contour lines
    contours = ax.contour(lon, lat, field, levels=10, colors='black', linewidths=0.5, 
                          alpha=0.6, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    # Add lat/lon gridlines
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, label="Bias (deg)", shrink=0.8)
    ax.set_title(title, fontsize=13, weight="bold")
    plt.show()

plot_field(bA, "Azimuth Bias Field (deg)")
plot_field(bE, "Elevation Bias Field (deg)")
