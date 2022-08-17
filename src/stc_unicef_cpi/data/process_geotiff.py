import glob
import os
import re
from pathlib import Path
from typing import List, Union

import cartopy.io.shapereader as shpreader
import dask.dataframe as dd
import geopandas as gpd
import h3.api.numpy_int as h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import swifter
from affine import Affine
from dask.distributed import Client, LocalCluster
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm.auto import tqdm

cluster = LocalCluster(
    scheduler_port=8786,
    n_workers=2,
    threads_per_worker=1,
    memory_limit="2GB",
)


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


def clip_tif_to_ctry(file_path, ctry_name, save_dir=None):
    """Clip a GeoTIFF to a specified country boundary,
    and write a new file, with .

    :param file_path: _description_
    :type file_path: _type_
    :param save_dir: _description_, defaults to None
    :type save_dir: _type_, optional
    :param ctry_name: _description_
    :type ctry_name: str, optional
    :raises ValueError: _description_
    """
    fname = Path(file_path).name
    # world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    world = reader.records()
    with rasterio.open(file_path, "r", masked=True) as tif_file:
        ctry_shp = next(
            filter(lambda x: x.attributes["NAME"] == ctry_name, world)
        ).geometry
        if tif_file.crs is not None and tif_file.crs != "EPSG:4326":
            # NB assumes that no CRS corresponds to EPSG:4326 (as standard)
            ctry_shp = gpd.GeoSeries(ctry_shp)
            ctry_shp.crs = "EPSG:4326"
            ctry_shp = ctry_shp.to_crs(tif_file.crs).geometry
        # world = world.to_crs(tif_file.crs)
        # ctry_shp = world[world.name == ctry_name].geometry
        try:
            out_image, out_transform = rasterio.mask.mask(tif_file, ctry_shp, crop=True)
        except TypeError:
            # polygon not iterable
            out_image, out_transform = rasterio.mask.mask(
                tif_file, [ctry_shp], crop=True
            )
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
        fname = ctry_name + "_" + fname
        with rasterio.open(save_dir / (fname.lower()), "w", **out_meta) as dest:
            dest.write(out_image)
    else:
        plt.imshow(out_image.squeeze())
        plt.show()
        # raise ValueError("Must specify save_dir")


def rxr_reproject_tiff_to_target(
    src_tiff_file: Union[str, Path],
    target_tiff_file: Union[str, Path],
    dest_path: Union[str, Path] = None,
    verbose=False,
):
    # f"shape: {raster.rio.shape}\n"
    # f"resolution: {raster.rio.resolution()}\n"
    # f"bounds: {raster.rio.bounds()}\n"
    # f"sum: {raster.sum().item()}\n"
    # f"CRS: {raster.rio.crs}\n"
    with rxr.open_rasterio(src_tiff_file, masked=True) as src_file, rxr.open_rasterio(
        target_tiff_file, masked=True
    ) as target_file:
        if src_file.rio.crs is None:
            src_file.rio.write_crs(4326, inplace=True)
            print("Warning, no CRS present in src, assuming EPSG:4326")
        # NB by default uses nearest-neighbour resampling
        # - simplest but often worse. Chosen bilinear here
        # instead as that's what GEE uses, so at least
        # now consistent for all data.
        # TODO: Consider if alternative resampling scheme
        # to bilinear is better for this task
        rxr_match = src_file.rio.reproject_match(
            target_file, resampling=rasterio.enums.Resampling(1)
        )
        if verbose:
            print_tif_metadata(rxr_match)
            print_tif_metadata(target_file)
            # fig, ax = plt.subplots()
            # rxr_match.plot(ax=ax)
            # fig2, ax2 = plt.subplots()
            # target_file.plot(ax=ax2)
        if dest_path is not None:
            dest_path = Path(dest_path)
            rxr_match.rio.to_raster(dest_path)
        else:
            return rxr_match


def geotiff_to_df(
    geotiff_filepath: str,
    spec_band_names: List[str] = None,
    max_bands=5,
    rm_prefix=None,
    verbose=False,
):
    """Convert a geotiff file to a pandas dataframe,
    and print some additional info.

    :param geotiff_filepath: path to a geotiff file
    :type geotiff_filepath: str
    :param spec_band_names: Specified band names - only used
                            if these are not specified in
                            the GeoTIFF itself, at which
                            point they are mandatory
    :type spec_band_names: List[str], optional
    :param verbose: verbose output, defaults to False
    :type verbose: bool, optional
    :returns: pandas dataframe
    """
    # # NB quadkeys are defined on Mercator projection, so must reproject
    # world = world.to_crs("EPSG:3395")  # world.to_crs(epsg=3395) would also work
    reproj = False
    with rxr.open_rasterio(geotiff_filepath, masked=True) as open_file:
        name = Path(geotiff_filepath).name
        if verbose:
            print_tif_metadata(open_file, name)
        if open_file.rio.crs != "EPSG:4326":
            og_proj = open_file.rio.crs
            if og_proj is None:
                print("Warning: no CRS found in geotiff file:")
                print("assuming EPSG:4326")
                open_file.rio.write_crs(4326, inplace=True)
            else:
                print("Reprojection to lat/lon required: completing...")
                reproj = True
                try:
                    if not open_file.crs.epsg_treats_as_latlong():
                        print("Warning")
                except:
                    print("Rasterio version available does not include epsg check:")
                    print("Warning: change of flipped crs")
        open_file = open_file.squeeze()
        open_file.name = "data"
        try:
            band_names = open_file.attrs["long_name"]
            if type(band_names) == str:
                # not really multiband but below still works
                band_names = [band_names]
            multi_bands = True
        except KeyError:
            try:
                assert len(open_file.shape) == 2
                print("Single band found only")
                multi_bands = False
                band_names = [
                    re.sub(rm_prefix, "", name)
                    .replace(".tif", "", 1)
                    .replace("Data", "", 1)
                ]
            except AssertionError:
                # Multi-Band but with different
                # naming convention
                with rasterio.open(geotiff_filepath) as rast_file:
                    if verbose:
                        print("rioxarray struggling...")
                        print("rasterio finds following metadata:")
                        print(rast_file.meta)
                    if rast_file.meta["count"] == 1:
                        print("Single band found only")
                        multi_bands = False
                        band_names = [
                            re.sub(rm_prefix, "", name)
                            .replace(".tif", "", 1)
                            .replace("Data", "", 1)
                        ]
                    else:
                        multi_bands = True
                        band_names = rast_file.descriptions
                        if set(band_names) == {None}:
                            try:
                                assert spec_band_names is not None
                            except AssertionError:
                                raise ValueError("Must specify band names")
                            try:
                                assert len(spec_band_names) == len(band_names)
                            except AssertionError:
                                raise ValueError(
                                    "Band names specified do not match number of bands"
                                )
                            band_names = spec_band_names
                        if verbose:
                            print("Found bands", band_names)
                            # print(rast_file.tags())
                            # print(rast_file.tags(1))
        # restrict to tifs of fewer than max bands
        # else can cause memory issues
        if len(band_names) > max_bands:
            raise ValueError(
                "Too many bands, will require a lot of RAM: instead advisable to use rast_to_agg_df"
            )
        else:
            df = open_file.to_dataframe()

    # print(df.reset_index().describe())
    df.drop(columns=["spatial_ref"], inplace=True)
    df.dropna(subset=["data"], inplace=True)
    if multi_bands:
        df.rename(
            index=dict(zip(range(1, len(band_names) + 1), band_names)), inplace=True
        )

    if len(df.index.names) == 3:
        # TO SORT FOR OTHER ORDERING
        assert df.index.names == ["band", "y", "x"]
        df.index.set_names(["band", "latitude", "longitude"], inplace=True)
    elif len(df.index.names) == 2:
        assert df.index.names == ["y", "x"]
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
        if nbands == 0:
            # undefined band names
            with rasterio.open(geotiff_filepath) as rast_file:
                nbands = len(rast_file.indexes)
        title = re.sub(rm_prefix, "", name).replace(".tif", "", 1)
        print(f"{nbands} bands found in {title}")
        if nbands > 1:
            raise ValueError("More than one band, need to handle")
        else:
            df.drop(columns=["band"], inplace=True)
        df.rename(columns={"data": title.replace("Data", "")}, inplace=True)
        df = df.reset_index()

    if reproj:
        # do reprojection of coords to lat/lon
        if verbose:
            print("REPROJECTING")
        transformer = Transformer.from_crs(og_proj, "EPSG:4326")
        coords = [
            transformer.transform(x, y) for x, y in df[["latitude", "longitude"]].values
        ]
        df[["latitude", "longitude"]] = coords

    df.dropna(inplace=True)
    # print(df.head())
    return df


def rast_to_agg_df(tiff_file, agg_fn=np.mean, resolution=7, max_bands=3, verbose=False):
    """Likely slower than using rioxarray fns, but
    benefit of handling groups of bands at a time, rather
    than all at once (v memory expensive) - only to be
    used for tiffs with many bands.

    :param tiff_file: _description_
    :type tiff_file: _type_
    :param agg_fn: _description_, defaults to np.mean
    :type agg_fn: _type_, optional
    :param resolution: _description_, defaults to 7
    :type resolution: int, optional
    :param max_bands: _description_, defaults to 3
    :type max_bands: int, optional
    :param verbose: be verbose, defaults to False
    :type verbose: bool, optional
    """
    with rasterio.open(tiff_file) as raster:
        band_names = np.array(raster.descriptions)
        nbands = len(band_names)
        ctr = 0

        if verbose:
            print("Finding pixel coords...")
        # get pixel coords
        T0 = raster.transform  # upper-left pixel corner affine transform
        # Get affine transform for pixel centres
        T1 = T0 * Affine.translation(0.5, 0.5)
        # Function to convert pixel row/column index (from 0) to easting/northing at centre
        rc2en = lambda r, c: T1 * (c, r)

        tmp = raster.read(1)
        # All rows and columns
        cols, rows = np.meshgrid(np.arange(tmp.shape[1]), np.arange(tmp.shape[0]))
        del tmp
        # All eastings and northings (there is probably a faster way to do this)
        eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows, cols)
        transformer = Transformer.from_crs(raster.crs, "EPSG:4326")
        lats, longs = transformer.transform(eastings, northings)
        del eastings, northings
        latlongs = np.dstack((lats, longs))
        if verbose:
            print("Calculating hex codes...")
        hex_codes = np.apply_along_axis(
            lambda x: h3.geo_to_h3(*x, resolution),
            axis=-1,
            arr=latlongs,
        )
        del latlongs
        # if verbose:
        #     print(f"Found {len(np.unique(hex_codes))} hex codes")
        res_df = None
        while ctr < nbands:
            band_idxs = np.array(
                list(set(range(ctr, ctr + max_bands)) & set(range(nbands))), dtype=int
            )
            if verbose:
                print(f"Done, now processing bands {band_names[band_idxs]}")
            if len(band_idxs) == 1:
                band_cols = band_names[band_idxs].tolist()
            else:
                band_cols = band_names[band_idxs].tolist()
            array = raster.read(list(band_idxs + 1))  # need + 1 as not zero-indexed
            # array = np.vstack((array, hex_codes[np.newaxis, ...]))
            df = pd.DataFrame(
                array.reshape([len(band_idxs), -1]).T,
                columns=band_cols,
            )
            df["hex_code"] = hex_codes.reshape(-1)
            df.dropna(how="all", inplace=True)
            del array
            if verbose:
                print("Aggregating...")
            df = df.groupby(by="hex_code").agg({band: agg_fn for band in band_cols})
            if res_df is None:
                res_df = df
            else:
                if verbose:
                    print("Joining to previous aggregations...")
                res_df = res_df.join(df, how="outer")
                del df
            ctr += max_bands
    res_df.dropna(how="all", inplace=True)
    return res_df


def agg_tif_to_df(
    df,
    tiff_dir,
    rm_prefix="cpi",
    agg_fn=np.mean,
    max_records=int(1e5),
    replace_old=True,
    resolution=7,
    verbose=False,
):
    """Pass df with hex_code column of numpy_int type h3 codes,
    and a directory with tiff files, then aggregate pixels from tiffs
    within each hexagon according to given function.

    Note that rather than using shapefiles, this uses pixel centroid
    values, hence different quantities of pixels may be aggregated
    in each hexagon, and it will not work sensibly at all if the
    resolution of the tiff file is lower than the resolution of
    the specified hexagons.

    :param df: 'ground truth' dataframe to aggregate tiffs to,
               with hex_code column at specified resolution
    :type df: pd.DataFrame
    :param tiff_dir: Either directory containing .tifs, a
                     single .tif file, or a list of .tif
                     files to aggregate to given df
    :type tiff_dir: _type_
    :param rm_prefix: Prefix to remove from file string when naming variables,
                      defaults to "cpi"
    :type rm_prefix: str, optional
    :param agg_fn: Function to use when aggregating tiff pixels within cells,
                   defaults to np.mean
    :type agg_fn: _type_, optional
    :param max_records: Max number of pixels in clipped tiff before using dask,
                        defaults to int(1e5)
    :type max_records: int, optional
    :param replace_old: Overwrite old columns if match new data,
                        defaults to True
    :type replace_old: bool, optional
    :param resolution: Resolution level of h3 grid to use, defaults to 7
    :type resolution: int, optional
    :param verbose: Verbose output, defaults to False
    :type verbose: bool, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    try:
        assert "hex_code" in df.columns
    except AttributeError:
        raise ValueError("hex_code not in df.columns")

    try:
        if os.path.isdir(tiff_dir):
            # absolute path to search for all tiff files inside a specified folder
            path = Path(tiff_dir) / "*.tif"
            tif_files = glob.glob(str(path))
        elif os.path.isfile(tiff_dir):
            tif_files = [tiff_dir]
    except TypeError:
        # list of tiff files passed directly
        assert type(tiff_dir) == list
        tif_files = tiff_dir

    for i, fname in enumerate(tif_files):
        title = re.sub(rm_prefix, "", Path(fname).name).replace(".tif", "", 1)
        print(f"Working with {title}: {i+1}/{len(tif_files)}...")
        # Convert to dataframe
        try:
            tmp = geotiff_to_df(fname, rm_prefix=rm_prefix, verbose=verbose)
        except:
            tmp = geotiff_to_df(
                fname,
                spec_band_names=["GDP_PPP_1990", "GDP_PPP_2000", "GDP_PPP_2015"],
                rm_prefix=rm_prefix,
                verbose=verbose,
            )
        print("Converted to dataframe!")
        if verbose:
            print("Dataframe info:")
            print(tmp.info())
        print("Adding hex info...")
        # need to split df into manageable chunks for memory
        if len(tmp.index) > max_records:
            print("Large dataframe, using dask instead...")
            try:
                with Client(
                    cluster, timeout="2s"
                ) as client:  # add options (?) e.g. n_workers=4, memory_limit="4GB"
                    # NB ideal to have partitions around 100MB in size
                    # client.restart()
                    ddf = dd.from_pandas(
                        tmp,
                        npartitions=max(
                            [4, int(tmp.memory_usage(deep=True).sum() // int(1e8))]
                        ),
                    )  # chunksize = max_records(?)
                    print(f"Using {ddf.npartitions} partitions")
                    ddf["hex_code"] = ddf[["latitude", "longitude"]].apply(
                        lambda row: h3.geo_to_h3(
                            row["latitude"], row["longitude"], resolution
                        ),
                        axis=1,
                        meta=(None, int),
                    )
                    ddf = ddf.drop(columns=["latitude", "longitude"])
                    print("Done!")
                    print("Aggregating within cells...")
                    ddf = ddf.groupby("hex_code").agg(
                        {col: agg_fn for col in ddf.columns if col != "hex_code"}
                    )
                    tmp = ddf.compute()
            except OSError:
                cluster = LocalCluster(
                    scheduler_port=8786,
                    n_workers=2,
                    threads_per_worker=1,
                    memory_limit="2GB",
                )
                with Client(
                    cluster,
                    timeout="2s",
                ) as client:  # add options (?) e.g. n_workers=4, memory_limit="4GB"
                    # client.restart()
                    # NB ideal to have partitions around 100MB in size
                    ddf = dd.from_pandas(
                        tmp,
                        npartitions=max(
                            [4, int(tmp.memory_usage(deep=True).sum() // int(1e8))]
                        ),
                    )  # chunksize = max_records(?)
                    print(f"Using {ddf.npartitions} partitions")
                    ddf["hex_code"] = ddf[["latitude", "longitude"]].apply(
                        lambda row: h3.geo_to_h3(
                            row["latitude"], row["longitude"], resolution
                        ),
                        axis=1,
                        meta=(None, int),
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
                lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], resolution),
                axis=1,
            )
            tmp.drop(columns=["latitude", "longitude"], inplace=True)
            print("Done!")
            print("Aggregating within cells...")
            tmp = tmp.groupby(by=["hex_code"]).agg(
                {col: agg_fn for col in tmp.columns if col != "hex_code"}
            )
        print("Joining to already aggregated data...")
        # Aggregate ground truth to hexagonal cells with mean
        # NB automatically excludes missing data for households,
        # so differing quantities of data for different values
        if replace_old:
            override_cols = [col for col in tmp.columns if col in df.columns]
            if len(override_cols) > 0:
                print("Overwriting old columns:", override_cols)
            df.drop(columns=override_cols, inplace=True)
        df = df.join(
            tmp,
            how="left",
            on="hex_code",
        )
        if verbose:
            print(
                "Non-nans in block after join:",
                len(df.dropna(subset=tmp.columns, how="all")),
            )
            # print(df.dropna(subset=tmp.columns).head())
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
        try:
            if not dataset.crs.epsg_treats_as_latlong():
                print("Warning")
        except:
            print("Rasterio version available does not include epsg check:")
            print("Warning: change of flipped crs")
            # TODO: Check handling transformations correctly
            # assignees: fitzgeraldja
            # labels: data, IMPORTANT
            # Important! Some bands suggest this is not the case,
            # but luckily not a huge problem as only transforming
            # one tiff currently.
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
    datasets: Union[List[str], List[bytes]],
    hex_codes: List[int],
    width=256,
    height=256,
    verbose=False,
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
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :return: _description_
    :rtype: _type_
    """
    nbands = 0
    for dataset in datasets:
        with rasterio.open(dataset) as open_file:
            nbands += len(open_file.indexes)
    if verbose:
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
    path = Path(tiff_dir) / "*.tif"
    # raw_path = path.encode("unicode_escape")
    tif_files = glob.glob(str(path))

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
