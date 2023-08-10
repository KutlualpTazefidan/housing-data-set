import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Library to visualize the missing the data
import missingno as msno

# Plot to identify poor neighborhoods
import altair as alt
import json

# Reprojecting geometrical data to adapt the coordinate system
import geopandas as gpd

# from vega_datasets import data as vega_data
df_kch = pd.read_csv("data/king_county_housing_details_a_sales.csv")
# Get info about the database: number of non null rows and data types
print(df_kch.info())
# Dimension of the dataset
print("Shape of the dataset:", np.shape(df_kch))
df_kch.head(5)
# converting the data type of date to date
df_kch["date"] = pd.to_datetime(df_kch["date"], format="%Y-%m-%d")
# Basic statistics analysis of the columns:
print(df_kch.describe())
print("Unique values for the waterfront:", df_kch["waterfront"].unique())
print("Unique values for the year renovated:", df_kch["yr_renovated"].unique())
print(
    "Unique values for the condition:",
    df_kch["condition"].min(),
    "-",
    df_kch["condition"].max(),
)
print("Unique values for the grade:", df_kch["grade"].min(), "-", df_kch["grade"].max())
msno.matrix(df_kch)
# check how many duplicated rows exist in the data frame
df_kch.duplicated().value_counts()
plt.rcParams.update(
    {"figure.figsize": (8, 5), "axes.facecolor": "white", "axes.edgecolor": "black"}
)
plt.rcParams["figure.facecolor"] = "w"
pd.plotting.register_matplotlib_converters()
pd.set_option("display.float_format", lambda x: "%.3f" % x)
fig, axs = plt.subplots(ncols=4, nrows=5, figsize=(20, 20))
index = 0
axs = axs.flatten()
for k, v in df_kch.items():
    if k != "date":
        sns.boxplot(y=k, data=df_kch, ax=axs[index])
        index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# Percentage of outliers
for key, value in df_kch.items():
    # q1 and q3: lower and upper quartiles
    q1 = value.quantile(0.25)
    q3 = value.quantile(0.75)
    # interuartile range
    ir = q3 - q1
    # only outliers
    v_col = value[(value <= q1 - 1.5 * ir) | (value >= q3 + 1.5 * ir)]
    # percentage of outliers, but waterfront,view, sqft_basement,yr_renovated has still missing values
    perc = np.shape(v_col)[0] * 100.0 / np.shape(df_kch)[0]
    print("Column %s outliers = %.2f%%" % (key, perc))
    # GeoJson for the houses:
houses_geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["long"], row["lat"]]},
            "properties": {"grade": row["grade"]},
        }
        for _, row in df_kch.iterrows()
    ],
}
houses_gdf = gpd.GeoDataFrame.from_features(houses_geojson_data)
# house_gdf
print(houses_gdf.columns)
houses_geojson_data["features"][0]["geometry"]["coordinates"]
houses_gdf.geometry[0].x
print(houses_gdf["geometry"].head())
# Load the GeoJSON data using geopandas
# web page: https://gis-kingcounty.opendata.arcgis.com/datasets/kingcounty::metropolitan-king-county-council-kccdst-area/explore?location=47.521009%2C-121.704444%2C8.98
# following contains nice example: https://github.com/altair-viz/altair/issues/588


# importing this allows to access the properties for the plot
# with open("data/Metropolitan_King_County_Council___kccdst_area.geojson") as f:
with open("data/Cities_and_Unincorporated_King_County___city_kc_area.geojson") as f:
    geojson_data = json.load(f)
geoData = alt.Data(values=geojson_data["features"])

print(geoData)
# Create an Altair chart
map_chart = (
    alt.Chart(geoData)
    .mark_geoshape(stroke="black", strokeWidth=0.5)
    .encode(
        color=alt.value("lightgray"),  # Set a default color
        tooltip=[
            alt.Tooltip(
                "properties.CITYNAME:O", title="City Name"
            ),  # Access nested property
        ],
    )
    .project(type="identity", reflectY=True)
    .properties(width=500, height=500)
)
# ).configure(background='transparent')
zoom = alt.selection_interval(bind="scales", encodings=["x", "y"])

interactive_map_chart = map_chart.add_selection(zoom)

interactive_map_chart


# Putting the x,y data into lat and long column; helped with the visualization
houses_gdf["latitude"] = houses_gdf["geometry"].apply(lambda geom: geom.y)
houses_gdf["longitude"] = houses_gdf["geometry"].apply(lambda geom: geom.x)

bin_size = 0.01  # Adjust the bin size as needed, to have a grid
houses_gdf["bin_longitude"] = houses_gdf["longitude"].apply(
    lambda lon: round(lon / bin_size) * bin_size
)
houses_gdf["bin_latitude"] = houses_gdf["latitude"].apply(
    lambda lat: round(lat / bin_size) * bin_size
)

# Specifying the ranges to match the scale of the maps
x_range = [houses_gdf["longitude"].min(), houses_gdf["longitude"].max()]
y_range = [houses_gdf["latitude"].min(), houses_gdf["latitude"].max()]
print("x&y range: ", x_range, y_range)
houses_gdf_subset = houses_gdf.head(1000)

houses_chart = (
    alt.Chart(houses_gdf)
    .mark_circle(color="red", size=100)
    .encode(
        longitude=alt.Longitude("longitude:Q"),
        latitude=alt.Latitude("latitude:Q"),
        tooltip=[alt.Tooltip("grade:O", title="Grade")],
        color=alt.Color(
            "grade:Q",
            scale=alt.Scale(scheme="viridis", domain=[0, 13]),
            legend=alt.Legend(
                direction="vertical",
                orient="right",
                title="Average Grade",
                titleOrient="top",
                cornerRadius=0.5,
            ),
        ),
    )
)


print(houses_gdf_subset.head(10))

grid_chart = (
    alt.Chart(houses_gdf)
    .transform_aggregate(
        count="count()",
        average_grade="average(grade)",
        groupby=["bin_longitude", "bin_latitude"],
    )
    .mark_circle(opacity=0.7)
    .encode(
        longitude="bin_longitude:Q",
        latitude="bin_latitude:Q",
        size=alt.Size("count:Q", title="Count", scale=alt.Scale(range=[1, 10])),
        color=alt.Color(
            "average_grade:Q", scale=alt.Scale(scheme="viridis", domain=[0, 13])
        ),
        tooltip=[
            alt.Tooltip("average_grade:Q", title="Average Grade"),
            alt.Tooltip("count:Q", title="Count"),
        ],
    )
    .properties(width=500, height=500)
)

# houses_gdff = gpd.GeoDataFrame(
#     houses_gdf,
#     geometry=gpd.points_from_xy(houses_gdf.longitude, houses_gdf.latitude),
#     crs=gdf.crs
# )

# Create a grid-based aggregation for houses
# grid_chart = alt.Chart(houses_gdf_subset).mark_rect(opacity=0.7).encode(
#     x=alt.X('longitude:Q', bin=alt.Bin(step=0.01),axis=None),
#     y=alt.Y('latitude:Q', bin=alt.Bin(step=0.01),axis=None),
#     # color=alt.Color('average(grade):Q', scale=alt.Scale(scheme='viridis',domain=[0, 13]),legend=alt.Legend(direction='vertical',orient='right', title='Average Grade',titleOrient='top',cornerRadius=0.5)),
# )

grid_chart.interactive()
# houses_chart

# Combine the map and house charts
alt.data_transformers.disable_max_rows()

combined_chart = map_chart + grid_chart
# combined_chart = map_chart + houses_chart

# Render the combined chart
# combined_chart.interactive()
