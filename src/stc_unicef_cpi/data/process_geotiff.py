import glob
import os
import re
from os import PathLike
from pathlib import Path
from typing import Callable, List, Optional, Pattern, Type, Union

import cartopy.io.shapereader as shpreader
import dask.dataframe as dd
import geopandas as gpd
import h3.api.numpy_int as h3
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
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
from xarray import DataArray, Dataset


def print_tif_metadata(
    rioxarray_rio_obj: Union[Dataset, DataArray, List[Dataset]],
    name: Optional[str] = None,
) -> None:
    """View metadata associated with a raster file,
    loaded using rioxarray

    :param rioxarray_rio_obj: rioxarray dataset object
    :type rioxarray_rio_obj: Union[Dataset, DataArray, List[Dataset]
    :param name: Name of tiff data, defaults to ""
    :type name: Optional[str], optional
    """
    if name is None:
        name = "your data"

    print(f"The crs of {name} is:", rioxarray_rio_obj.rio.crs)  # type: ignore
    print(f"The nodatavalue of {name} is:", rioxarray_rio_obj.rio.nodata)  # type: ignore
    print(f"The shape of {name} is:", rioxarray_rio_obj.shape)  # type: ignore
    print(f"The spatial resolution for {name} is:", rioxarray_rio_obj.rio.resolution())  # type: ignore
    print(f"The metadata for {name} is:", rioxarray_rio_obj.attrs)  # type: ignore


def clip_tif_to_ctry(
    file_path: Union[PathLike, str],
    ctry_name: str,
    save_dir: Optional[Union[PathLike, str]] = None,
) -> None:
    """Clip a GeoTIFF to a specified country boundary,
    and write a new file to the specified directory if given,
    else just plot the clipped tiff. File name is prepended
    with the country name.

    :param file_path: Path to file to clip
    :type file_path: Union[PathLike,str]
    :param ctry_name: Name of country to clip to
    :type ctry_name: str
    :param save_dir: Path to directory to save to, defaults to None (just plot)
    :type save_dir: Optional[Union[PathLike,str]], optional
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
    return None


def rxr_reproject_tiff_to_target(
    src_tiff_file: Union[str, PathLike],
    target_tiff_file: Union[str, PathLike],
    dest_path: Optional[Union[str, PathLike]] = None,
    verbose: bool = False,
) -> Union[Dataset, DataArray, List[Dataset], None]:
    """Use rioxarray and an example (target) tiff to
    reproject the given (source) tiff to the same CRS
    and resolution.

    :param src_tiff_file: Path to tiff file you want to reproject
    :type src_tiff_file: Union[str, PathLike]
    :param target_tiff_file: Path to tiff file that is example of desired projection and resolution
    :type target_tiff_file: Union[str, PathLike]
    :param dest_path: Path to write reprojected tiff to, defaults to None (just return reprojected raster)
    :type dest_path: Optional[Union[str, PathLike]], optional
    :param verbose: Verbosity, defaults to False
    :type verbose: bool, optional
    :return: Either None (if dest_path is not None) or reprojected raster
    :rtype:Union[Dataset, DataArray, List[Dataset], None]
    """
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
            return None
        else:
            return rxr_match


def geotiff_to_df(
    geotiff_filepath: Union[str, PathLike],
    spec_band_names: Optional[List[str]] = None,
    max_bands: int = 5,
    rm_prefix: Union[str, Pattern[str]] = "",
    verbose: bool = False,
) -> pd.DataFrame:
    """Convert a geotiff file to a pandas dataframe,
    and print some additional info.

    :param geotiff_filepath: path to a geotiff file
    :type geotiff_filepath: Union[str, PathLike]
    :param spec_band_names: Specified band names - only used
                            if these are not specified in
                            the GeoTIFF itself, at which
                            point they are mandatory, defaults to None
    :type spec_band_names: Optional[List[str]], optional
    :param max_bands: Max allowable bands before requires use of rast_to_agg_df, defaults to 5
    :type max_bands: Optional[int], optional
    :param rm_prefix: Prefix (or regex pattern) to replace in file name, defaults to None
    :type rm_prefix: Union[str, Pattern[str]], optional
    :param verbose: verbose output, defaults to False
    :type verbose: bool, optional
    :raises ValueError: No band names provided but none found either
    :raises ValueError: Number of band names provided when none found does not match number of bands
    :raises ValueError: Too many bands to handle without excessive memory - use rast_to_agg_df instead
    :raises ValueError: Problem with index resulting from conversion to df
    :raises ValueError: Problem converting bands
    :return: pandas dataframe of lat, long, val for each band
    :rtype: pd.DataFrame
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


def rast_to_agg_df(
    tiff_file: Union[str, Path, bytes],
    agg_fn: Callable[[npt.NDArray], npt.NDArray] = np.mean,
    resolution: int = 7,
    max_bands: int = 3,
    verbose: bool = False,
) -> pd.DataFrame:
    """Likely slower than using rioxarray fns, but
    benefit of handling groups of bands at a time, rather
    than all at once (v memory expensive) - only to be
    used for tiffs with many bands.

    :param tiff_file: Path to (many banded, large) tiff file
    :type tiff_file: Union[str, PathLike]
    :param agg_fn: Aggregation function, defaults to np.mean
    :type agg_fn: Callable[[npt.NDArray], npt.NDArray], optional
    :param resolution: Resolution of H3 grid to aggregate to, defaults to 7
    :type resolution: int, optional
    :param max_bands: Max number of bands to process at one time, defaults to 3
    :type max_bands: int, optional
    :param verbose: Verbose, defaults to False
    :type verbose: bool, optional
    :return: Dataframe of aggregated data
    :rtype: pd.DataFrame
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
        longs, lats = transformer.transform(eastings, northings)
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
    res_df.dropna(how="all", inplace=True)  # type: ignore
    return res_df


def agg_tif_to_df(
    df: pd.DataFrame,
    tiff_dir: Union[str, PathLike, List[str], List[PathLike]],
    rm_prefix: Union[str, Pattern[str]] = "cpi",
    agg_fn: Callable[[npt.NDArray], npt.NDArray] = np.mean,
    max_records: int = int(1e5),
    replace_old: bool = True,
    resolution: int = 7,
    verbose: bool = False,
) -> pd.DataFrame:
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
    :type tiff_dir: Union[str, PathLike]
    :param rm_prefix: Prefix or regex pattern to remove from file string when naming variables,
                      defaults to "cpi"
    :type rm_prefix: Union[str, Pattern[str]], optional
    :param agg_fn: Function to use when aggregating tiff pixels within cells,
                   defaults to np.mean
    :type agg_fn: Callable[[npt.NDArray], npt.NDArray], optional
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
    :raises ValueError: hex_code column not in df
    :return: Original dataframe with new columns added from aggregated values of tiffs in hexes
    :rtype: pd.DataFrame
    """
    try:
        assert "hex_code" in df.columns
    except AttributeError:
        raise ValueError("hex_code not in df.columns")

    try:
        if os.path.isdir(tiff_dir):  # type: ignore
            # absolute path to search for all tiff files inside a specified folder
            path = Path(tiff_dir) / "*.tif"  # type: ignore
            tif_files = glob.glob(str(path))
        elif os.path.isfile(tiff_dir):  # type: ignore
            tif_files = [tiff_dir]  # type: ignore
    except TypeError:
        # list of tiff files passed directly
        assert type(tiff_dir) == list
        tif_files = tiff_dir  # type: ignore

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
                    "dask-scheduler:8786", timeout="2s"
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
                try:
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
                except RuntimeError:
                    # scheduler for dask failed to start, just use numpy + pandas
                    print("dask failed, switching back to numpy + pandas")
                    tmp["hex_code"] = np.apply_along_axis(
                        lambda row: h3.geo_to_h3(*row, 7),
                        arr=tmp[["latitude", "longitude"]].values,
                        axis=1,
                    )
                    tmp["hex_code"] = tmp.hex_code.astype(
                        int
                    )  # ensure converted to int
                    tmp.drop(columns=["latitude", "longitude"], inplace=True)
                    print("Done!")
                    print("Aggregating within cells...")
                    tmp = tmp.groupby(by=["hex_code"]).agg(
                        {col: agg_fn for col in tmp.columns if col != "hex_code"}
                    )

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
    dataset: Union[Dataset, DataArray, List[Dataset], rasterio.io.DatasetReader],
    lat: float,
    long: float,
    dim_x: int = 256,
    dim_y: int = 256,
    verbose: bool = False,
) -> npt.NDArray:
    """Extract an array of specified dimensions (num pixels) about
    specified lat/long - centered by default

    :param dataset: rioxarray or rasterio dataset (open tiff file)
    :type dataset: Union[Dataset, DataArray, List[Dataset], rasterio.io.DatasetReader]
    :param lat: Latitude of center point about which to extract image
    :type lat: float
    :param long: Longitude of center point about which to extract image
    :type long: float
    :param dim_x: x dimension (pixel width) of extracted image, defaults to 256
    :type dim_x: int, optional
    :param dim_y: y dimension (pixel height) of extracted image, defaults to 256
    :type dim_y: int, optional
    :param verbose: Verbose, defaults to False
    :type verbose: bool, optional
    :return: Array of tiff values ('image') at specified coordinates, of given size
    :rtype: npt.NDArray
    """
    try:
        og_proj = dataset.crs  # type: ignore
        assert og_proj is not None
    except AttributeError:
        print("No crs attribute, assuming EPSG:4326")
        og_proj = "EPSG:4326"
    except AssertionError:
        print("No crs specified, assuming EPSG:4326")
        og_proj = "EPSG:4326"
    if og_proj != "EPSG:4326" and og_proj is not None:
        if verbose:
            print("WARNING, tiff not in lat/long")
        # reproject lat/lon given to tiff crs
        transformer = Transformer.from_crs("EPSG:4326", og_proj)
        try:
            if not dataset.crs.epsg_treats_as_latlong():  # type: ignore
                print("Warning")
        except:
            print("Rasterio version available does not include epsg check:")
            print("Warning: change of flipped crs")
            # TODO: Check handling transformations correctly, and check again if lat long transformation correct
            # assignees: fitzgeraldja
            # labels: data, IMPORTANT
            # Important! Some bands suggest this is not the case,
            # but luckily not a huge problem as only transforming
            # one tiff currently.
        lat, long = transformer.transform(lat, long)
    # TODO: Add try, except block for when out of bounds error thrown
    row, col = dataset.index(long, lat)  # type: ignore
    max_i, max_j = dataset.height, dataset.width  # type: ignore
    left = col - dim_x // 2
    top = row - dim_y // 2
    window = Window(left, top, dim_x, dim_y)
    subset = dataset.read(window=window)  # type: ignore
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
    datasets: Union[List[str], List[PathLike]],
    hex_codes: Union[List[int], npt.NDArray[np.int_]],
    width: int = 256,
    height: int = 256,
    verbose: bool = False,
) -> npt.NDArray:
    """For a set of datasets, specified by file path, and
    a set of h3 hex codes, extract centered
    images of specified size and return a 4D array
    in shape
        (image_idx,band,i,j).

    :param datasets: List of paths to tiff files for which you want to extract (and stack) 'image' bands
    :type datasets: Union[List[str], List[PathLike]]
    :param hex_codes: Set of H3 hex codes in numpy_int format for which you wish to extract images
    :type hex_codes: Union[List[int], npt.NDArray[np.int_]]
    :param width: Width of extracted images in pixels, defaults to 256
    :type width: int, optional
    :param height: Height of extracted images in pixels, defaults to 256
    :type height: int, optional
    :param verbose: Verbose, defaults to False
    :type verbose: bool, optional
    :return: Extracted images in shape (image_idx,band,i,j)
    :rtype: npt.NDArray
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
    tiff_dir: Union[str, PathLike],
    hex_codes: Union[List[int], npt.NDArray[np.int_]],
    dim_x: int = 256,
    dim_y: int = 256,
) -> npt.NDArray:
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
    :type tiff_dir: Union[str, PathLike]
    :param hex_codes: Set of H3 hex codes for which you wish to extract images
    :type hex_codes: Union[List[int], npt.NDArray[np.int_]]
    :param dim_x: Pixel width of extracted images, defaults to 256
    :type dim_x: int, optional
    :param dim_y: Pixel height of extracted images, defaults to 256
    :type dim_y: int, optional
    :return: Array of images at hex coords, in shape (hex_id, band, i, j)
    :rtype: npt.NDArray
    """
    # absolute path to search for all tiff files inside a specified folder
    path = Path(tiff_dir) / "*.tif"
    # raw_path = path.encode("unicode_escape")
    tif_files = glob.glob(str(path))

    all_ims = extract_ims_from_hex_codes(tif_files, hex_codes, dim_x, dim_y)
    return all_ims


def resample_tif(
    tif_file_path: Union[str, PathLike],
    dest_dir: Union[str, PathLike],
    rescale_factor: Optional[float] = 2.0,
) -> None:
    """Resample a tiff file by a given factor, using bilinear resampling
    - greater than 1 corresponds to increased resolution,
    less than 1 decreased.

    :param tif_file_path: Path to tiff file to resample
    :type tif_file_path: Union[str, PathLike]
    :param dest_dir: Destination directory for resampled tiff file to be written to
    :type dest_dir: Union[str, PathLike]
    :param rescale_factor: Rescale factor, defaults to 2
    :type rescale_factor: Optional[int], optional
    """
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
