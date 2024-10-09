#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pyproj import Proj, transform
import numpy as np

#%%

path='/home/llaurabat/my-spatial-smi-oldv/data/'
loc_file = f'coarsen_all_items_loc.csv'

### Loading Data
loc_df = pd.read_csv(path + loc_file)
floating_id = np.genfromtxt(path + 'floating_profiles_id.txt', dtype=int, delimiter=',')
#%%
loc_df['floating'] = loc_df['LP'].isin(floating_id)
loc_df['Easting'] = loc_df['Easting']*1000
loc_df['Northing'] = loc_df['Northing']*1000
loc_df_sub = loc_df[['Easting', 'Northing']].copy()
#%%
test_LPs = [84, 88, 133, 139, 307, 363, 446, 448, 472, 544, 617, 770, 814, 1125,
 1134, 1142, 1205, 1302, 1339, 1341]
val_LPs = [83, 104, 138, 294, 301, 348, 377, 441, 732, 1132, 1198, 1199, 1204, 1301,
 1327, 1329, 1330, 1345, 1348]


#%%
v84 = Proj(proj="latlong",towgs84="0,0,0",ellps="WGS84")
v36 = Proj(proj="latlong", k=0.9996012717, ellps="airy",
        towgs84="446.448,-125.157,542.060,0.1502,0.2470,0.8421,-20.4894")
vgrid = Proj(init="world:bng")
#%%


def vectorized_convert(df):
    vlon36, vlat36 = vgrid(df['Easting'].values, 
                           df['Northing'].values, 
                           inverse=True)
    converted = transform(v36, v84, vlon36, vlat36)
    df['longitude'] = converted[0]
    df['latitude'] = converted[1]
    return df

# df = pd.DataFrame({'northing': [378778, 384732],
#                    'easting': [366746, 364758]})

print(vectorized_convert(loc_df))
#%%
colors = np.where(loc_df['floating'], 'blue', 'red')
colors_val = np.where(loc_df['LP'].isin(val_LPs), 'blue', 'white')
colors_test = np.where(loc_df['LP'].isin(test_LPs), 'blue', 'white')

#%%
# Sample longitude and latitude data
# Replace these with your actual data
# longitude = [-0.1278, -1.2577, -3.1883, -5.9301]  # Example longitudes
# latitude = [51.5074, 52.4862, 55.9533, 57.1497]  # Example latitudes

# Create a plot with a specific map projection
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-10, 2, 49, 61], crs=ccrs.PlateCarree())  # Set extent to cover the UK

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Plot scatter points
plt.scatter(loc_df.longitude, loc_df.latitude, color=colors_test, marker='o', s=0.5, transform=ccrs.Geodetic())

# Add gridlines (optional)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

plt.savefig('debug.png')
# Show the plot
plt.show()
# %%
