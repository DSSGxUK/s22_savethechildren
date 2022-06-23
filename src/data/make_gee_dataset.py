import datetime
import logging
import urllib.request
from pathlib import Path

import click
import matplotlib.pyplot as plt
import rasterio
import requests  # type: ignore
import rioxarray as rxr
from dotenv import find_dotenv, load_dotenv
from humanfriendly import format_size
from rasterio.plot import show
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


# import ee
# import ee.mapclient

# # Initialize the Earth Engine module.
# ee.Initialize()

# # (Example) Print metadata for a DEM dataset.
# # print(ee.Image("USGS/SRTMGL1_003").getInfo())

# # import unconstrained worldpop estimates on 100m grid - see here for distinction
# # https://www.worldpop.org/methods/top_down_constrained_vs_unconstrained/
# # Unconstrained chosen as better for smaller rural pops + temporal variation
# unconstrained_worldpop = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")

# # Initial date of interest (inclusive).
# i_date = "2020-01-01"

# # Final date of interest (exclusive).
# f_date = "2021-01-01"

# # Selection of appropriate bands and dates for worldpop.

# ee.mapclient.centerMap(-110, 40, 5)

# # Filter to only include images within the colorado and utah boundaries.
# polygon = ee.Geometry.Polygon(
#     [
#         [
#             [-109.05, 37.0],
#             [-102.05, 37.0],
#             [-102.05, 41.0],  # colorado
#             [-109.05, 41.0],
#             [-111.05, 41.0],
#             [-111.05, 42.0],  # utah
#             [-114.05, 42.0],
#             [-114.05, 37.0],
#             [-109.05, 37.0],
#         ]
#     ]
# )

# unconstrained_worldpop = unconstrained_worldpop.select("population").filterBounds(
#     polygon
# )
# # .filterDate(
# #     i_date, f_date
# # )

# # Select the median pixel.
# image1 = unconstrained_worldpop.median()

# # Select specific bands.
# # image = image1.select("M_5", "W_5", "M_20")
# ee.mapclient.addToMap(image1,
#                     #   {"gain": [1.4, 1.4, 1.1]}
#                       )
# Create a Landsat 7 composite for Spring of 2000, and filter by
# the bounds of the FeatureCollection.

import datetime
import webbrowser

import ee
import folium

# ee.Authenticate()

ee.Initialize()

# Import the MODIS land surface temperature collection.
lst = ee.ImageCollection("MODIS/006/MOD11A1")
# Initial date of interest (inclusive).
i_date = "2017-01-01"

# Final date of interest (exclusive).
f_date = "2020-01-01"

# Selection of appropriate bands and dates for LST.
lst = lst.select("LST_Day_1km", "QC_Day").filterDate(i_date, f_date)

# Define the urban location of interest as a point near Lyon, France.
u_lon = 4.8148
u_lat = 45.7758
u_poi = ee.Geometry.Point(u_lon, u_lat)

# Define the rural location of interest as a point away from the city.
r_lon = 5.175964
r_lat = 45.574064
r_poi = ee.Geometry.Point(r_lon, r_lat)

scale = 500  # scale in meters
# Calculate and print the mean value of the LST collection at the point.
lst_urban_point = lst.mean().sample(u_poi, scale).first().get("LST_Day_1km").getInfo()
print(
    "Average daytime LST at urban point:",
    round(lst_urban_point * 0.02 - 273.15, 2),
    "°C",
)

# Get the data for the pixel intersecting the point in urban area.
lst_u_poi = lst.getRegion(u_poi, scale).getInfo()

# Get the data for the pixel intersecting the point in rural area.
lst_r_poi = lst.getRegion(r_poi, scale).getInfo()

# Preview the result.
print(lst_u_poi[:5])

import pandas as pd


def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # print(df.head())

    # Remove rows without data inside.
    df = df[["longitude", "latitude", "time", *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors="coerce")

    # Convert the time field into a datetime.
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")

    # Keep the columns of interest.
    df = df[["time", "datetime", *list_of_bands]]

    return df


lst_df_urban = ee_array_to_df(lst_u_poi, ["LST_Day_1km"])


def t_modis_to_celsius(t_modis):
    """Converts MODIS LST units to degrees Celsius."""
    t_celsius = 0.02 * t_modis - 273.15
    return t_celsius


# Apply the function to get temperature in celsius.
lst_df_urban["LST_Day_1km"] = lst_df_urban["LST_Day_1km"].apply(t_modis_to_celsius)

# Do the same for the rural point.
lst_df_rural = ee_array_to_df(lst_r_poi, ["LST_Day_1km"])
lst_df_rural["LST_Day_1km"] = lst_df_rural["LST_Day_1km"].apply(t_modis_to_celsius)

print(lst_df_urban.head())

# Define a region of interest with a buffer zone of 1000 km around Lyon.
roi = u_poi.buffer(1e6)

# Reduce the LST collection by mean.
lst_img = lst.mean()

# Adjust for scale factor.
lst_img = lst_img.select("LST_Day_1km").multiply(0.02)

# Convert Kelvin to Celsius.
lst_img = lst_img.select("LST_Day_1km").add(-273.15)

# print("Projection, crs, and crs_transform:", lst_img.projection())
# print("Scale in meters:", lst_img.projection().nominalScale())

# Create a URL to the styled image for a region around France.
url = lst_img.getThumbUrl(
    {
        "min": 10,
        "max": 30,
        "dimensions": 512,
        "region": roi,
        "palette": ["blue", "yellow", "orange", "red"],
    }
)
print(url)

# Display the thumbnail land surface temperature in France.
# print("\nPlease wait while the thumbnail loads, it may take a moment...")
# webbrowser.open(url)

# Get a feature collection of administrative boundaries.
countries = ee.FeatureCollection("FAO/GAUL/2015/level0").select("ADM0_NAME")

# Filter the feature collection to subset France.
france = countries.filter(ee.Filter.eq("ADM0_NAME", "France"))

# Clip the image by France.
lst_fr = lst_img.clip(france)

# Create a buffer zone of 10 km around Lyon.
lyon = u_poi.buffer(10000)  # meters

# Create the URL associated with the styled image data.
url = lst_fr.getThumbUrl(
    {
        "min": 0,
        "max": 2500,
        "region": roi,
        "dimensions": 512,
        "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"],
    }
)
print(url)

# task = ee.batch.Export.image.toDrive(
#     image=lst_img,
#     description="land_surf_temp_near_lyon_france",
#     scale=30,
#     region=lyon,
#     fileNamePrefix="my_export_lyon",
#     crs="EPSG:4326",
#     fileFormat="GeoTIFF",
# )
# task.start()

link = lst_img.getDownloadURL(
    {"scale": 30, "crs": "EPSG:3395", "fileFormat": "GeoTIFF", "region": lyon}
)
print(link)


def estim_file_size(url):
    with requests.get(url, stream=True) as file_get:
        estim_file_size = int(file_get.headers["Content-Length"])
    return format_size(estim_file_size)


def print_tif_metadata(rioxarray_rio_obj):
    # View metadata associated with the raster file
    print("The crs of your data is:", rioxarray_rio_obj.rio.crs)
    print("The nodatavalue of your data is:", rioxarray_rio_obj.rio.nodata)
    print("The shape of your data is:", rioxarray_rio_obj.shape)
    print(
        "The spatial resolution for your data is:", rioxarray_rio_obj.rio.resolution()
    )
    print("The metadata for your data is:", rioxarray_rio_obj.attrs)


download_data = False
if download_data:
    print(f"Estimated file size: {estim_file_size(link)}")
    download_url(link, "../../data/test.zip")

    import zipfile

    with zipfile.ZipFile("../../data/test.zip", "r") as f:
        f.extractall("../../data/test")

with rxr.open_rasterio("../../data/test/download.LST_Day_1km.tif", masked=True) as src:
    # # NB quadkeys are defined on Mercator projection, so must reproject
    # world = world.to_crs("EPSG:3395")  # world.to_crs(epsg=3395) would also work

    print_tif_metadata(src)
    print(src.coords)

    # for i, shape in enumerate(src.block_shapes, 1):
    #     print((i, shape))
    df = src.squeeze()
    df.name = "data"
    df = df.to_dataframe()
    print(df.head())
    # fig, ax = plt.subplots(dpi=200)
    # src.astype("int").plot()
    # # # transform rasterio plot to real world coords
    # # extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]
    # # show(src, extent=extent, ax=ax)
    # plt.show()

import quadkey

# example extraction of quadkey from lat lon
print(quadkey.lonlat2quadint(4.8148, 45.7758))
# example extraction of lat lon bounding points at given level from quadkey (found from lat lon)
print(quadkey.tile2bbox(quadkey.lonlat2quadint(4.8148, 45.7758), 14))


# except Exception:
raise Exception


def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds a method for displaying Earth Engine image tiles to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True,
    ).add_to(self)


# Add Earth Engine drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

# # import json

# # with open("./worldpop_data_info.json", "w") as f:
# #     json.dump(unconstrained_worldpop.getInfo(), f)


# # def download_climate_gee_data():
# #     pass

# # @click.command()
# # @click.argument("input_filepath", type=click.Path(exists=True))
# # @click.argument("output_filepath", type=click.Path())
# # def main(input_filepath, output_filepath):
# #     """Runs data processing scripts to turn raw data from (../raw) into
# #     cleaned data ready to be analyzed (saved in ../processed).
# #     """
# #     logger = logging.getLogger(__name__)
# #     logger.info("making final data set from raw data")


# # if __name__ == "__main__":
# #     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# #     logging.basicConfig(level=logging.INFO, format=log_fmt)

# #     # not used in this stub but often useful for finding various files
# #     project_dir = Path(__file__).resolve().parents[2]

# #     # find .env automagically by walking up directories until it's found, then
# #     # load up the .env entries as environment variables
# #     load_dotenv(find_dotenv())

# #     main()
