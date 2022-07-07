import glob
from pathlib import Path
from typing import Union

import h3.api.numpy_int as h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
from pyproj import Transformer
from rasterio.windows import Window
from tqdm.auto import tqdm


def print_tif_metadata(rioxarray_rio_obj, name=""):
    """View metadata associated with a raster file,
    loaded using rioxarray

    :param rioxarray_rio_obj: rioxarray object
    :type rioxarray_rio_obj: rioxarray.rio.Dataset
    :param name: Name of tiff data
    :type name: str, optional

    """
    if len(name) == 0:
        name = "your data"

    print(f"The crs of {name} is:", rioxarray_rio_obj.rio.crs)
    print(f"The nodatavalue of {name} is:", rioxarray_rio_obj.rio.nodata)
    print(f"The shape of {name} is:", rioxarray_rio_obj.shape)
    print(f"The spatial resolution for {name} is:", rioxarray_rio_obj.rio.resolution())
    print(f"The metadata for {name} is:", rioxarray_rio_obj.attrs)


def geotiff_to_df(geotiff_filepath: str):
    """Convert a geotiff file to a pandas dataframe,
    and print some additional info.

    :param geotiff_filepath: path to a geotiff file
    :type geotiff_filepath: str
    :returns: pandas dataframe
    """
    # # NB quadkeys are defined on Mercator projection, so must reproject
    # world = world.to_crs("EPSG:3395")  # world.to_crs(epsg=3395) would also work
    reproj = False
    with rxr.open_rasterio(geotiff_filepath, masked=True) as open_file:
        name = Path(geotiff_filepath).name
        print_tif_metadata(open_file, name)
        if open_file.rio.crs != "EPSG:4326":
            print("Reprojection to lat/lon required: completing...")
            reproj = True
            og_proj = open_file.rio.crs
        open_file = open_file.squeeze()
        open_file.name = "data"
        band_names = open_file.attrs["long_name"]
        df = open_file.to_dataframe()
    # print(df.reset_index().describe())
    df.drop(columns=["spatial_ref"], inplace=True)
    df.dropna(subset=["data"], inplace=True)
    df.rename(index=dict(zip(range(1, len(band_names) + 1), band_names)), inplace=True)

    if len(df.index.names) == 3:
        df.index.set_names(["band", "latitude", "longitude"], inplace=True)
    elif len(df.index.names) == 2:
        df.index.set_names(["latitude", "longitude"], inplace=True)
    else:
        raise ValueError("Unexpected number of index levels")
    # Now converted to csv with band/lat/lon index
    # So unstack band index into separate columns if multband,
    # and drop unhelpful column multiindex
    if len(df.index.names) == 3:
        df = df.unstack(level=0).droplevel(0, axis=1).reset_index()
    else:
        # Single band, but check to make sure, then rename data
        # suitably
        nbands = df.band.nunique()
        title = name.lstrip("cpi").rstrip(".tif")
        print(f"{nbands} bands found in {title}")
        if nbands > 1:
            raise ValueError("More than one band, need to handle")
        else:
            df.drop(columns=["band"], inplace=True)
        df.rename(columns={"data": title.strip("Data")}, inplace=True)
        df = df.reset_index()

    if reproj:
        # do reprojection of coords to lat/lon
        transformer = Transformer.from_crs(og_proj, "EPSG:4326")
        coords = [
            transformer.transform(x, y) for x, y in df[["latitude", "longitude"]].values
        ]
        df[["latitude", "longitude"]] = coords

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


def extract_image_at_coords(
    dataset, lat: float, long: float, dim_x=256, dim_y=256, verbose=False
):
    """Extract an array of specified dimensions (num pixels) about
    specified lat/long - centered by default

    :param dataset: _description_
    :type dataset: rxr.rio.Dataset or rasterio dataset
    :param lat: _description_
    :type lat: float
    :param long: _description_
    :type long: float
    :param dim_x: _description_
    :type dim_x: int
    :param dim_y: _description_
    :type dim_y: int
    :param centered: _description_, defaults to True
    :type centered: bool, optional
    """

    og_proj = dataset.crs
    if og_proj != "EPSG:4326":
        if verbose:
            print("WARNING, tiff not in lat/long")
        # reproject lat/lon given to tiff crs
        transformer = Transformer.from_crs("EPSG:4326", og_proj)
        lat, long = transformer.transform(lat, long)
    row, col = dataset.index(long, lat)
    max_i, max_j = dataset.height, dataset.width
    left = col - dim_x // 2
    top = row - dim_y // 2
    window = Window(left, top, dim_x, dim_y)
    subset = dataset.read(window=window)
    return subset


# Example ^
# lat, long = h3.h3_to_geo(609534210041970687)
# all_ims = None
# with rasterio.open(
#     "/Users/johnf/Downloads/raw_low_res_dssg/cpiSlopeData.tif", masked=True
# ) as open_file:
#     windowed_im = extract_image_at_coords(open_file, lat, long, 256, 256)
#     print(np.isnan(windowed_im).sum() / np.prod(windowed_im.shape))
#     # plt.imshow(windowed_im[0,:,:])
#     # plt.show()
#     if all_ims is None:
#         all_ims = windowed_im
#     else:
#         all_ims = np.vstack((all_ims, windowed_im))


def extract_ims_from_hex_codes(
    datasets: Union[list[str], list[bytes]], hex_codes: list[int], width=256, height=256
):
    """For a set of datasets, specified by file path, and
    a set of h3 hex codes, extract centered
    images of specified size and return a 4D array
    in shape
        (image_idx,band,i,j).


    :param datasets: _description_
    :type datasets: list[str]
    :param hex_codes: _description_
    :type hex_codes: list[int]
    :param width: _description_, defaults to 256
    :type width: int, optional
    :param height: _description_, defaults to 256
    :type height: int, optional
    :return: _description_
    :rtype: _type_
    """
    nbands = 0
    for dataset in datasets:
        with rasterio.open(dataset) as open_file:
            nbands += len(open_file.indexes)
    print(f"Overall, {nbands} bands found in datasets")
    ims = np.zeros((len(hex_codes), nbands, height, width))
    latlongs = [h3.h3_to_geo(hex_code) for hex_code in hex_codes]
    running_nband = 0
    for ds_idx, dataset in tqdm(
        enumerate(datasets), position=0, total=len(datasets), desc="Dataset progress:"
    ):
        with rasterio.open(dataset, masked=True) as open_file:
            ds_nbands = len(open_file.indexes)
            for idx, latlong in tqdm(
                enumerate(latlongs),
                position=1,
                total=len(latlongs),
                desc="Location progress:",
            ):
                im = extract_image_at_coords(
                    open_file, *latlong, dim_x=width, dim_y=height
                )  # type: ignore
                if im.shape[1:] != (height, width):
                    # NB know that must be less than, as
                    # trying to read window of specified size
                    y_diff = height - im.shape[1]
                    x_diff = width - im.shape[2]
                    im = np.pad(
                        im,
                        (
                            (0, 0),
                            (y_diff // 2, y_diff // 2 + y_diff % 2),
                            (x_diff // 2, x_diff // 2 + x_diff % 2),
                        ),
                        "constant",
                    )
                ims[idx, running_nband : running_nband + ds_nbands, :, :] = im
            running_nband += ds_nbands
    return ims


def convert_tiffs_to_image_dataset(
    tiff_dir: str, hex_codes: list[int], dim_x=256, dim_y=256
) -> np.ndarray:
    """Convert set of GeoTIFFs to a 4D numpy array according
    to specified dataset - expect the path to a directory
    containing all relevant GeoTIFFs with extension '.tif',
    and a list of h3 hexagon identifiers in numpy_int form
    (use import h3.api.numpy_int as h3).

    Returned array is in form (hex_id, band, i, j), with
    i, j through the band image array defaulting to size
    256 x 256, as specified by dim_x, dim_y.

    :param tiff_dir: Path to GeoTIFF directory, with file
                     extensions '.tif'
    :type tiff_dir: str
    :param df_file_path: _description_
    :type df_file_path: str
    :param dim_x: _description_, defaults to 256
    :type dim_x: int, optional
    :param dim_y: _description_, defaults to 256
    :type dim_y: int, optional
    :return: _description_
    :rtype: np.ndarray
    """
    # absolute path to search for all tiff files inside a specified folder
    path = tiff_dir.rstrip("/") + "/*.tif"
    # raw_path = path.encode("unicode_escape")
    tif_files = glob.glob(path)

    all_ims = extract_ims_from_hex_codes(tif_files, hex_codes, dim_x, dim_y)
    return all_ims
