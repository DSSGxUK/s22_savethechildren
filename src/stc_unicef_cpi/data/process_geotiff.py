import glob
from pathlib import Path
from typing import List, Union

import dask.dataframe as dd
import geopandas as gpd
import h3.api.numpy_int as h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import swifter
from dask.distributed import Client
from pyproj import Transformer
from rasterio.enums import Resampling
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


def clip_tif_to_ctry(file_path, save_dir=None, ctry_name="Nigeria"):
    """Clip a GeoTIFF to a specified country boundary,
    and write a new file, with .

    :param file_path: _description_
    :type file_path: _type_
    :param save_dir: _description_, defaults to None
    :type save_dir: _type_, optional
    :param ctry_name: _description_, defaults to "Nigeria"
    :type ctry_name: str, optional
    :raises ValueError: _description_
    """
    fname = Path(file_path).name
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    with rasterio.open(file_path, "r", masked=True) as tif_file:
        world = world.to_crs(tif_file.crs)
        ctry_shp = world[world.name == ctry_name].geometry
        out_image, out_transform = rasterio.mask.mask(tif_file, ctry_shp, crop=True)
        out_meta = tif_file.meta

    if save_dir is not None:
        save_dir = Path(save_dir)
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rasterio.open(
            save_dir / (ctry_name + "_" + fname), "w", **out_meta
        ) as dest:
            dest.write(out_image)
    else:
        raise ValueError("Must specify save_dir")


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
        try:
            band_names = open_file.attrs["long_name"]
            multi_bands = True
        except KeyError:
            assert len(open_file.shape) == 2
            print("Single band found only")
            multi_bands = False
        df = open_file.to_dataframe()
    # print(df.reset_index().describe())
    df.drop(columns=["spatial_ref"], inplace=True)
    df.dropna(subset=["data"], inplace=True)
    if multi_bands:
        df.rename(
            index=dict(zip(range(1, len(band_names) + 1), band_names)), inplace=True
        )

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


def agg_tif_to_df(
    df, tiff_dir, rm_prefix="cpi", agg_fn=np.mean, max_records=int(1e5), verbose=False
):
    """Pass df with hex_code column of numpy_int type h3 codes,
    and a directory with tiff files, then aggregate pixels from tiffs
    within each hexagon according to given function.

    Note that rather than using shapefiles, this uses pixel centroid
    values, hence different quantities of pixels may be aggregated
    in each hexagon, and it will not work sensibly at all if the
    resolution of the tiff file is lower than the resolution of
    the specified hexagons.

    :param df: _description_
    :type df: _type_
    :param tiff_dir: _description_
    :type tiff_dir: _type_
    :param rm_prefix: _description_, defaults to "cpi"
    :type rm_prefix: str, optional
    :param agg_fn: _description_, defaults to np.mean
    :type agg_fn: _type_, optional
    :param max_records: _description_, defaults to int(1e5)
    :type max_records: _type_, optional
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    try:
        assert "hex_code" in df.columns
    except AttributeError:
        raise ValueError("hex_code not in df.columns")

    # absolute path to search for all tiff files inside a specified folder
    path = tiff_dir.rstrip("/") + "/*.tif"
    tif_files = glob.glob(path)
    for i, fname in enumerate(tif_files):
        title = Path(fname).name.lstrip(rm_prefix).rstrip(".tif")
        print(f"Working with {title}: {i+1}/{len(tif_files)}...")
        # Convert to dataframe
        tmp = geotiff_to_df(fname)
        print("Converted to dataframe!")
        if verbose:
            print("Dataframe info:")
            print(tmp.info())
        print("Adding hex info...")
        # need to split df into manageable chunks for memory
        if len(tmp.index) > max_records:
            print("Large dataframe, using dask instead...")
            with Client() as client:  # add options (?) e.g. n_workers=4, memory_limit="4GB"
                # NB ideal to have partitions around 100MB in size
                ddf = dd.from_pandas(
                    tmp, npartitions=len(tmp) // max_records + 1
                )  # chunksize = max_records(?)

                ddf["hex_code"] = ddf[["latitude", "longitude"]].apply(
                    lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], 7),
                    axis=1,
                    meta=(None, "int64"),
                )
                ddf = ddf.drop(columns=["latitude", "longitude"])
                print("Done!")
                print("Aggregating within cells...")
                ddf = ddf.groupby("hex_code").agg(
                    {col: agg_fn for col in ddf.columns if col != "hex_code"}
                )
                tmp = ddf.compute()
        else:
            tmp["hex_code"] = tmp[["latitude", "longitude"]].swifter.apply(
                lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], 7), axis=1
            )
            tmp.drop(columns=["latitude", "longitude"], inplace=True)
            print("Done!")
            print("Aggregating within cells...")
            tmp = tmp.groupby(by=["hex_code"], as_index=False).agg(
                {col: agg_fn for col in tmp.columns if col != "hex_code"}
            )
        print("Joining to survey data...")
        # Aggregate ground truth to hexagonal cells with mean
        # NB automatically excludes missing data for households,
        # so differing quantities of data for different values
        df = df.merge(
            tmp,
            how="left",
            on="hex_code",
        )
        print("Done!")
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
        if dataset.crs.epsg_treats_as_latlong():
            # TODO: Check handling transformations correctly
            # assignees: fitzgeraldja
            # labels: data, IMPORTANT
            # Important! Some bands suggest this is not the case,
            # but luckily not a huge problem as only transforming
            # one tiff currently.
            pass
        lat, long = transformer.transform(lat, long)
    # TODO: Add try, except block for when out of bounds error thrown
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
    datasets: Union[List[str], List[bytes]], hex_codes: List[int], width=256, height=256
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
    tiff_dir: str, hex_codes: List[int], dim_x=256, dim_y=256
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


def resample_tif(tif_file_path, dest_dir, rescale_factor=2):
    with rasterio.open(tif_file_path) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * rescale_factor),
                int(dataset.width * rescale_factor),
            ),
            resampling=Resampling.bilinear,
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
        )
        dest_dir = Path(dest_dir)
        name = Path(tif_file_path).name
        with rasterio.open(
            dest_dir / name,
            "w",
            driver="GTiff",
            height=data.shape[-2],
            width=data.shape[-1],
            count=dataset.count,
            dtype=data.dtype,
            crs=dataset.crs,
            transform=transform,
        ) as dest:
            dest.write(data)
