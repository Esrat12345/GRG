import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Step 1: Load the .pos Data File
# Adjust the path as needed
#pos_data = pd.read_csv('SIH1740_Dataset/Dataset1.pos', delim_whitespace=True, header=None)



# Load the .pos file and process the data
try:
    # Read the .pos file, assuming the structure is consistent with your description
    pos_data = pd.read_csv(
        'SIH1740_Dataset/Dataset1.pos',
        delim_whitespace=True,
        comment='%',  # Skips lines that start with '%'
        header=None,
        usecols=[0, 1, 2, 3],  # Only select relevant columns
    )
    pos_data.columns = ['Date', 'Time', 'Latitude', 'Longitude']

    # Combine date and time into a single datetime column
    pos_data['Datetime'] = pd.to_datetime(pos_data['Date'] + ' ' + pos_data['Time'])
    pos_data.drop(columns=['Date', 'Time'], inplace=True)  # Remove the original columns
    pos_data = pos_data[['Datetime', 'Latitude', 'Longitude']]  # Reorder columns for clarity

    # Create geometry for geopandas using longitude and latitude
    pos_data['geometry'] = pos_data.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    gdf = gpd.GeoDataFrame(pos_data, geometry='geometry')

    # Calculate bounding box coordinates and validate
    minx, miny, maxx, maxy = gdf.total_bounds
    if None in (minx, miny, maxx, maxy):
        raise ValueError("Bounding box coordinates contain None values.")

    # Generate the road network graph within the bounding box using OSMnx
    G = ox.graph_from_bbox(miny, maxy, minx, maxx, network_type='all')  # Set network_type as needed

    # Plot the road network with the vehicle path
    fig, ax = ox.plot_graph(G, show=False, close=False)
    gdf.plot(ax=ax, color='red', marker='o', markersize=5, label='Vehicle Path')  # Vehicle path in red
    ax.legend()
    plt.show()

except pd.errors.ParserError as e:
    print("Error reading the .pos file:", e)
except ValueError as e:
    print("Value error:", e)
except KeyError as e:
    print("Key error:", e)
except Exception as e:
    print("An unexpected error occurred:", e)


# Step 3: Classify Movement and Road Type Detection
# Shift latitude, longitude, and time to calculate distances and speeds
pos_data['Shifted_Lat'] = pos_data['Latitude'].shift(-1)
pos_data['Shifted_Lon'] = pos_data['Longitude'].shift(-1)
pos_data['Shifted_Time'] = pos_data['Time'].shift(-1)

# Define Haversine function for distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Calculate distances and speeds
pos_data['Distance'] = pos_data.apply(lambda row: haversine(row['Latitude'], row['Longitude'], row['Shifted_Lat'], row['Shifted_Lon']), axis=1)
pos_data['Speed'] = pos_data['Distance'] / (pos_data['Shifted_Time'] - pos_data['Time']).dt.total_seconds()

# KMeans clustering to classify segments based on speed
kmeans = KMeans(n_clusters=2)
pos_data = pos_data.dropna(subset=['Speed'])  # Drop any rows with NaN values for clustering
pos_data['Road_Type'] = kmeans.fit_predict(pos_data[['Speed']])

# Map clusters to road types based on speed (manual labeling if necessary)
road_type_labels = {0: 'Service Road', 1: 'Main Road'}  # Adjust if needed
pos_data['Road_Type'] = pos_data['Road_Type'].map(road_type_labels)

# Display classification results
print(pos_data[['Latitude', 'Longitude', 'Speed', 'Road_Type']].head())

# Step 4: Calculate Distance on Main Road
# Filter for main road segments and sum distances
main_road_distance = pos_data[pos_data['Road_Type'] == 'Main Road']['Distance'].sum()
print(f"Total distance traveled on the main road: {main_road_distance:.2f} km")
