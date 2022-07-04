from pathlib import Path

import pandas as pd
import rioxarray as rxr


def print_tif_metadata(rioxarray_rio_obj, name=""):
    """View metadata associated with a raster file,
    loaded using rioxarray

    :param rioxarray_rio_obj: rioxarray object
    :type rioxarray_rio_obj: rioxarray.rio.Dataset
    """
    if len(name) == 0:
        name = "your data"

    print(f"The crs of {name} is:", rioxarray_rio_obj.rio.crs)
    print(f"The nodatavalue of {name} is:", rioxarray_rio_obj.rio.nodata)
    print(f"The shape of {name} is:", rioxarray_rio_obj.shape)
    print(f"The spatial resolution for {name} is:", rioxarray_rio_obj.rio.resolution())
    print(f"The metadata for {name} is:", rioxarray_rio_obj.attrs)


def geotiff_to_df(geotiff_filepath):
    """Convert a geotiff file to a pandas dataframe,
    and print some additional info.

    :param geotiff_filepath: path to a geotiff file
    :type geotiff_filepath: str
    :returns: pandas dataframe
    """
    # # NB quadkeys are defined on Mercator projection, so must reproject
    # world = world.to_crs("EPSG:3395")  # world.to_crs(epsg=3395) would also work
    with rxr.open_rasterio(geotiff_filepath, masked=True) as open_file:
        name = Path(geotiff_filepath).name
        print_tif_metadata(open_file, name)
        open_file = open_file.squeeze()
        open_file.name = "data"
        band_names = open_file.attrs["long_name"]
        df = open_file.to_dataframe()
    # print(df.reset_index().describe())
    df.drop(columns=["spatial_ref"], inplace=True)
    df.dropna(subset=["data"], inplace=True)
    df.rename(index=dict(zip(range(1, len(band_names) + 1), band_names)), inplace=True)
    df.index.set_names(["band", "latitude", "longitude"], inplace=True)
    # print(df.head())
    return df


# Example
# geotiff_to_df("/Users/johnf/Downloads/cpiCopLandData.tiff")
# with rxr.open_rasterio(
#     "/Users/johnf/Downloads/cpiCopLandData.tiff", masked=True
# ) as src:

#     print_tif_metadata(src)
#     print(src.coords)

#     # for i, shape in enumerate(src.block_shapes, 1):
#     #     print((i, shape))
#     df = src.squeeze()
#     df.name = "data"
#     df = df.to_dataframe()
#     print(df.head())
#     print(df.info())
#     print(df.describe())
#     print(df.reset_index().head())
#     print(df.reset_index().info())
#     print(df.reset_index().describe())
#     # fig, ax = plt.subplots(dpi=200)
#     # src.astype("int").plot()
#     # # # transform rasterio plot to real world coords
#     # # extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]
#     # # show(src, extent=extent, ax=ax)
#     # plt.show()
