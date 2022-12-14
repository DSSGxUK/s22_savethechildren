import argparse
import glob as glob
import logging
import os
import sys
import warnings
from functools import partial, reduce
from pathlib import Path

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import h3.api.numpy_int as h3
import numpy as np
import pandas as pd
import pycountry
import rasterio
import rioxarray as rxr
import shapely.wkt
import tensorflow as tf
from art import tprint
from rich import print

import stc_unicef_cpi.data.get_osm_data as osm
import stc_unicef_cpi.data.process_geotiff as pg
import stc_unicef_cpi.data.process_netcdf as net
import stc_unicef_cpi.utils.constants as c
import stc_unicef_cpi.utils.general as g
import stc_unicef_cpi.utils.geospatial as geo
from stc_unicef_cpi.data.stream_data import RunStreamer

try:
    from stc_unicef_cpi.features import get_autoencoder_features as gaf
except ModuleNotFoundError:
    warnings.warn(
        "Necessary module for autoencoder features not found: assuming not desired"
    )


def read_input_unicef(path_read) -> pd.DataFrame:
    """Read source data provided by STC and UNICEF
    :param path_read: path to read data from
    :type path_read: str
    :return: database with target variable
    :rtype: dataframe
    """
    df = pd.read_csv(path_read, low_memory=False)
    return df


def select_country(df, country_code, lat, long) -> pd.DataFrame:
    """Select country of interest
    :param df: input provided by UNICEF and STC
    :type df: dataframe
    :param country_code: country code
    :type country_code: str
    :param lat: colname containing latitude measures
    :type lat: numerical
    :param long: colname containing longitude measures
    :type long: numerical
    :return: database with info related to country of interest
    :rtype: dataframe
    """
    df.columns = df.columns.str.lower()
    subset = df[df["countrycode"].str.strip() == country_code].copy()
    subset.dropna(subset=[lat, long], inplace=True)
    return subset


def aggregate_dataset(df) -> pd.DataFrame:
    """Aggregate dataset
    :param df: input required to aggregate
    :type df: dataframe
    :return: agg mean, agg count
    :rtype: dataframe, dataframe
    """
    df_mean = df.groupby(by=["hex_code"], as_index=False).mean()
    df_count = df.groupby(by=["hex_code"], as_index=False).count()[
        ["hex_code", "survey"]
    ]
    return df_mean, df_count


def create_target_variable(
    country_code, res, lat, long, threshold, read_dir, copy_to_nbrs=False
) -> pd.DataFrame:
    """Create target variable
    :param country_code: country code related to country of interest
    :type country_code: str
    :param res: resolution of country of interest
    :type res: int
    :param lat: latitude of country of interest
    :type lat: numeric
    :param long: longitude of country of interest
    :type long: numeric
    :param threshold: minimal number of surveys per hexagon
    :type threshold: int
    :param read_dir: directory from where to read dataset
    :type read_dir: str
    :param copy_to_nbrs: include neighbouring resolution, defaults to False
    :type copy_to_nbrs: bool, optional
    :raises ValueError: no raw survey data available
    :return: dataset with observations satisfying conditions
    :rtype: dataframe
    """
    try:
        source = Path(read_dir) / "childpoverty_microdata_gps_21jun22.csv"
    except FileNotFoundError:
        raise ValueError(f"Must have raw survey data available in {read_dir}")
    df = read_input_unicef(source)
    sub = select_country(df, country_code, lat, long)
    try:
        assert len(sub) > 0
    except AssertionError:
        raise ValueError(
            f"No geocoded data available in given dataset for {pycountry.countries.get(alpha_3=country_code).name}"
        )
    # Create variables for two or more deprivations
    for k in range(2, 5):
        sub[f"dep_{k}_or_more_sev"] = sub["sumpoor_sev"] >= k
    sub = geo.get_hex_code(sub, lat, long, res)
    sub = sub.reset_index(drop=True)
    if copy_to_nbrs:
        sub["hex_incl_nbrs"] = sub[["location", "hex_code"]].apply(
            lambda row: h3.k_ring(row["hex_code"], 1)
            if row["location"] == 1
            else h3.k_ring(row["hex_code"], 2),
            axis=1,
        )
        sev_cols = [col for col in sub.columns if "_sev" in col]
        other_cols = [
            col
            for col in sub.columns
            if ("int" in str(sub[col].dtype) or "float" in str(sub[col].dtype))
        ]
        agg_dict = {col: "mean" for col in other_cols}
        agg_dict.update({idx: ["mean", "count"] for idx in sev_cols})
        sub = sub.explode("hex_incl_nbrs").groupby(by=["hex_incl_nbrs"]).agg(agg_dict)
        sub.columns = ["_".join(col) for col in sub.columns.values]
        sub.rename(
            columns={
                f"{sev}_mean": f"{sev.replace('dep_','').replace('_sev','')}_prev"
                for sev in sev_cols
                if sev != "deprived_sev"
            },
            inplace=True,
        )
        sub.rename(
            columns={
                f"{sev}_count": f"{sev.replace('dep_','').replace('_sev','')}_count"
                for sev in sev_cols
                if sev != "deprived_sev"
            },
            inplace=True,
        )
        sub.drop(columns=["hex_code_mean"], inplace=True)
        survey_threshold = sub[sub.sumpoor_count >= threshold].reset_index().copy()
        survey_threshold.rename(columns={"hex_incl_nbrs": "hex_code"}, inplace=True)
        survey_threshold = geo.get_hex_centroid(survey_threshold, "hex_code")
    else:
        sub_mean, sub_count = aggregate_dataset(sub)
        sub_count = sub_count[sub_count.survey >= threshold]
        survey = geo.get_hex_centroid(sub_mean, "hex_code")
        survey_threshold = sub_count.merge(survey, how="left", on="hex_code")
    if copy_to_nbrs:
        print(
            f" -- After expanding, have {len(survey_threshold)} hexes with >= {threshold} surveys"
        )
    else:
        print(f" -- Found {len(survey_threshold)} hexes with >= {threshold} surveys")
    return survey_threshold


def change_name_reproject_tiff(
    tiff, attribute, country, read_dir=c.ext_data, out_dir=c.int_data
) -> None:
    """Rename attributes and reprojection of Tiff file
    :param tiff: path to tiff file
    :type tiff: str
    :param attributes: attributes names
    :type attributes: list of lists
    :param country: contry of interest
    :type country: str
    :param read_dir: path to read external data from, defaults to c.ext_data
    :type read_dir: str, optional
    """
    with rxr.open_rasterio(tiff) as data:
        fname = Path(tiff).name
        data.attrs["long_name"] = attribute
        data.rio.to_raster(tiff)
        try:
            gee_dir = Path(read_dir) / "gee"
            assert gee_dir.exists()
        except AssertionError:
            raise FileNotFoundError(
                f"Must have GEE data available in {gee_dir} - currently must manually download there from Google Drive."
            )
    p_r = Path(read_dir) / "gee" / f"cpi_poptotal_{country.lower()}_500.tif"
    pg.rxr_reproject_tiff_to_target(tiff, p_r, Path(out_dir) / fname, verbose=True)


@g.timing
def preprocessed_tiff_files(
    country, read_dir=c.ext_data, out_dir=c.int_data, force=False
) -> None:
    """Preprocess tiff files

    :param country: country of interest
    :type country: str
    :param read_dir: path to read data from, defaults to c.ext_data
    :type read_dir: str, optional
    :param out_dir: path to save data, defaults to c.int_data
    :type out_dir: str, optional
    :param force: force clipping, defaults to False
    :type force: bool, optional
    """
    g.create_folder(out_dir)
    # clip gdp ppp 30 arc sec
    print(" -- Clipping gdp pp 30 arc sec")
    if not (Path(read_dir) / (country.lower() + "_gdp_ppp_30.tif")).exists() or force:
        net.netcdf_to_clipped_array(
            Path(read_dir) / "gdp_ppp_30.nc", ctry_name=country, save_dir=read_dir
        )

    # clip ec and gdp
    print(" -- Clipping ec and gdp")
    tifs = glob.glob(str(Path(read_dir) / "*" / "*" / "2019" / "*.tif"))
    if (
        not all(
            [
                (
                    Path(read_dir) / (country.lower() + "_" + str(Path(tif).name))
                ).exists()
                for tif in tifs
            ]
        )
        or force
    ):
        partial_func = partial(
            pg.clip_tif_to_ctry, ctry_name=country, save_dir=read_dir
        )
        list(map(partial_func, tifs))

    # reproject resolution + crs
    print(" -- Reprojecting resolution & determining crs")
    econ_tiffs = sorted(glob.glob(str(Path(read_dir) / f"{country.lower()}_*.tif")))
    econ_tiffs = [ele for ele in econ_tiffs if "africa" not in ele]
    if (
        not all([(Path(out_dir) / Path(fname).name).exists() for fname in econ_tiffs])
        or force
    ):
        attributes = [
            ["gdp_2019"],
            ["ec_2019"],
            ["gdp_ppp_1990", "gdp_ppp_2000", "gdp_ppp_2015"],
        ]
        mapfunc = partial(change_name_reproject_tiff, country=country)
        list(map(mapfunc, econ_tiffs, attributes))

    # critical infrastructure data
    print(" -- Reprojecting critical infrastructure data")
    cisi_ctry = Path(read_dir) / f"{country.lower()}_africa.tif"
    fname = Path(cisi_ctry).name
    if not (Path(out_dir) / fname).exists() or force:
        cisi = glob.glob(str(Path(read_dir) / "*" / "*" / "010_degree" / "africa.tif"))[
            0
        ]
        pg.clip_tif_to_ctry(cisi, ctry_name=country, save_dir=read_dir)
        p_r = Path(read_dir) / "gee" / f"cpi_poptotal_{country.lower()}_500.tif"
        pg.rxr_reproject_tiff_to_target(
            cisi_ctry, p_r, Path(out_dir) / fname, verbose=True
        )


@g.timing
def preprocessed_speed_test(speed, res, country) -> pd.DataFrame:
    """Processing speed test data
    :param speed: dataset containing speed test data
    :type speed: dataframe
    :param res: grid resolution
    :type res: int
    :param country: country of interest
    :type country: str
    :return: clipped speed data to country, reprojected and aggregated
    :rtype: dataframe
    """
    logging.info("Clipping speed data to country - can take a couple of mins...")
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    world = reader.records()
    country = pycountry.countries.search_fuzzy(args.country)[0]
    ctry_name = country.name
    ctry_geom = next(
        filter(lambda x: x.attributes["NAME"] == ctry_name, world)
    ).geometry
    # now use low res to roughly clip
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ctry = world[world.name == ctry_name]
    bd_series = speed.geometry.str.replace(r"POLYGON\s\(+|\)", "").str.split(r"\s|,\s")
    speed["min_x"] = bd_series.str[0].astype("float")
    speed["max_y"] = bd_series.str[-1].astype("float")
    minx, miny, maxx, maxy = ctry.bounds.values.T.squeeze()
    # use rough bounds to restrict more or less to country
    speed = speed[
        speed.min_x.between(minx - 1e-1, maxx + 1e-1)
        & speed.max_y.between(miny - 1e-1, maxy + 1e-1)
    ].copy()
    speed["geometry"] = speed.geometry.swifter.apply(shapely.wkt.loads)
    speed = gpd.GeoDataFrame(speed, crs="epsg:4326")
    # only now look for intersection, as expensive
    try:
        ctry_geom = gpd.GeoDataFrame(ctry_geom, columns=["geometry"], crs="EPSG:4326")
    except ValueError:
        # problem for single geometry
        ctry_geom = gpd.GeoDataFrame([ctry_geom], columns=["geometry"], crs="EPSG:4326")
    speed = gpd.sjoin(speed, ctry_geom, how="inner", op="intersects").reset_index(
        drop=True
    )
    tmp = speed.geometry.swifter.apply(
        lambda x: pd.Series(np.array(x.centroid.coords.xy).flatten())
    )
    speed[["long", "lat"]] = tmp
    speed = geo.get_hex_code(speed, "lat", "long", res)
    speed = (
        speed[["hex_code", "avg_d_kbps", "avg_u_kbps"]]
        .groupby("hex_code")
        .mean()
        .reset_index()
    )
    return speed


@g.timing
def preprocessed_commuting_zones(country, res, read_dir=c.ext_data) -> pd.DataFrame:
    """Preprocess commuting zones
    :param country: country of interest
    :type country: str
    :param res: grid resolution
    :type res: int
    :param read_dir: path to read data from, defaults to c.ext_data
    :type read_dir: str, optional
    :return: processed information of commuting zones
    :rtype: dataframe
    """
    commuting = pd.read_csv(Path(read_dir) / "commuting_zones.csv", low_memory=False)
    commuting = commuting[commuting["country"] == country]
    comm = list(commuting["geometry"])
    comm_zones = pd.concat(list(map(partial(geo.hexes_poly, res=res), comm)))
    comm_zones = comm_zones.merge(commuting, on="geometry", how="left")
    comm_zones = comm_zones.add_suffix("_commuting")
    comm_zones.rename(columns={"hex_code_commuting": "hex_code"}, inplace=True)
    return comm_zones


@g.timing
def append_features_to_hexes(
    country,
    res,
    encoders,
    gpu,
    force=False,
    force_download=False,
    audience=False,
    read_dir=c.ext_data,
    save_dir=c.int_data,
    model_dir=c.base_dir_model,
    tiff_dir=c.tiff_data,
    hyper_tuning=False,
) -> pd.DataFrame:
    """Append features to hexagons withing a country
    :param country: country of interest
    :type country: str
    :param res: grid resolution
    :type res: int
    :param encoders: whether to append autoencoder features
    :type encoders: bool
    :param gpu: whether to use gpus or not
    :type gpu: bool
    :param force: force clipping, defaults to False
    :type force: bool, optional
    :param force_download: force download, defaults to False
    :type force_download: bool, optional
    :param audience: whether or not to include audience estimates, defaults to False
    :type audience: bool, optional
    :param read_dir: path to read input, defaults to c.ext_data
    :type read_dir: str, optional
    :param save_dir: path to save output, defaults to c.int_data
    :type save_dir: str, optional
    :param model_dir: path to model, defaults to c.base_dir_model
    :type model_dir: str, optional
    :param tiff_dir: path to tiff files, defaults to c.tiff_data
    :type tiff_dir: str, optional
    :param hyper_tuning: whether or not to perform hyperparameter tuning, defaults to False
    :type hyper_tuning: bool, optional
    :return: hexes with corresponding features
    :rtype: dataframe
    """
    # Setting up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
    file_handler = logging.FileHandler(c.dataset_log)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Starting process...")

    # Retrieve external data
    print(
        f"Initiating data retrieval. Audience: {audience}. Forced data recreation: {force}. Forced redownload: {force_download}."
    )
    RunStreamer(
        country,
        res,
        force_download,
        audience,
        read_path=str(c.ext_data),
        name_logger=c.str_log,
    )
    logger.info("Finished data retrieval.")
    logger.info(
        f"Please check your 'gee' folder in google drive and download all content to {read_dir}/gee. May take some time to appear."
    )
    logger.info("Check https://code.earthengine.google.com/tasks to monitor progress.")

    # Country hexes
    logger.info(f"Retrieving hexagons for {country} at resolution {res}.")
    hexes_ctry = geo.get_hexes_for_ctry(country, res)
    # expand by 2 hexes to ensure covers all data
    outer_hexes = geo.get_new_nbrs_at_k(hexes_ctry, 2)
    hexes_ctry = np.concatenate((hexes_ctry, outer_hexes), dtype=int)
    ctry = pd.DataFrame(hexes_ctry, columns=["hex_code"])
    # ctry.to_csv("./tmp_ctry.csv", index=False)  # temporary file for debugging

    # Facebook connectivity metrics
    if audience:
        logger.info(
            f"Collecting audience estimates for {country} at resolution {res}..."
        )
        fb = pd.read_parquet(
            Path(read_dir) / f"fb_aud_{country.lower()}_res{res}.parquet"
        )
        fb = geo.get_hex_centroid(fb)

    # Preprocessed tiff files
    logger.info(f"Preprocessing tiff files from {read_dir} and saving to {save_dir}..")
    preprocessed_tiff_files(country, read_dir, save_dir, force=force)

    # Conflict Zones
    logger.info("Reading and computing conflict zone estimates...")
    cz = pd.read_csv(Path(read_dir) / "conflict/GEDEvent_v22_1.csv", low_memory=False)
    cz = cz[cz.country == country]
    cz = geo.get_hex_code(cz, "latitude", "longitude", res)
    cz = geo.create_geometry(cz, "latitude", "longitude")
    cz = geo.aggregate_hexagon(cz, "geometry", "n_conflicts", "count")

    # Commuting zones
    logger.info("Reading and computing commuting zone estimates...")
    commuting = preprocessed_commuting_zones(country, res, read_dir)[c.cols_commuting]

    # Economic data
    logger.info("Retrieving features from economic tif files...")
    econ_files = glob.glob(str(Path(save_dir) / f"{country.lower()}*.tif"))

    econ = pg.agg_tif_to_df(
        ctry,
        econ_files,
        resolution=res,
        rm_prefix=rf"cpi|_|{country.lower()}|500",
        verbose=True,
    )

    # Google Earth Engine
    logger.info("Retrieving features from google earth engine tif files...")
    gee_files = glob.glob(str(Path(read_dir) / "gee" / f"*_{country.lower()}*.tif"))
    max_bands = 3
    gee_nbands = np.zeros(len(gee_files))
    for idx, file in enumerate(gee_files):
        with rasterio.open(file) as tif:
            gee_nbands[idx] = tif.count
    small_gee = np.array(gee_files)[gee_nbands < max_bands]
    large_gee = np.array(gee_files)[gee_nbands >= max_bands]
    gee = pg.agg_tif_to_df(
        ctry,
        list(small_gee),
        resolution=res,
        rm_prefix=rf"cpi|_|{country.lower()}|500",
        verbose=True,
    )

    for large_file in large_gee:
        large_gee_df = pg.rast_to_agg_df(
            large_file, resolution=res, max_bands=max_bands, verbose=True
        )
        gee = gee.join(
            large_gee_df,
            on="hex_code",
            how="outer",
        )

    # Join GEE with Econ
    logger.info("Merging aggregated features from tiff files to hexagons...")
    # econ.to_csv(Path(save_dir) / f"tmp_{country.lower()}_econ.csv")
    # gee.to_csv(Path(save_dir) / f"tmp_{country.lower()}_gee.csv")

    images = gee.merge(econ, on=["hex_code"], how="outer")
    del econ

    # Road density
    if force:
        logger.info("Recreating road density estimates...")
        road_file_name = "road_density_" + country.lower() + "_res" + str(res) + ".csv"
        rd = osm.get_road_density(country, res)
        rd.to_csv(Path(read_dir) / road_file_name, index=False)
    logger.info("Reading road density estimates...")
    road = pd.read_csv(Path(read_dir) / f"road_density_{country.lower()}_res{res}.csv")

    # Speed Test
    logger.info("Reading speed test estimates...")
    speed = pd.read_csv(
        Path(read_dir) / "connectivity" / "2021-10-01_performance_mobile_tiles.csv"
    )
    speed = preprocessed_speed_test(speed, res, country)

    # Open Cell Data
    logger.info("Reading open cell data...")
    cell = g.read_csv_gzip(
        glob.glob(str(Path(read_dir) / f"{country.lower().replace(' ','_')}_*gz.tmp"))[
            0
        ]
    )
    cell = geo.get_hex_code(cell, "lat", "long", res)
    cell = cell[["hex_code", "radio", "avg_signal"]]
    cell = (
        cell.groupby(by=["hex_code", "radio"])
        .size()
        .unstack(level=1)
        .fillna(0)
        .join(cell.groupby("hex_code").avg_signal.mean())
    ).reset_index()

    # Collected Data
    logger.info("Merging all features")
    dfs = [ctry, commuting, cz, road, speed, cell, images]
    assert all(["hex_code" in df.columns for df in dfs])
    hexes = reduce(
        lambda left, right: pd.merge(left, right, on="hex_code", how="left"), dfs
    )

    # Get autoencoders
    if encoders:
        tiffs = Path(tiff_dir / country.lower())
        tiffs.mkdir(exist_ok=True)
        print("--- Copying tiff files to tiff directory")
        gaf.copy_files(c.ext_data / "gee", tiffs, country.lower())
        gaf.copy_files(c.ext_data, tiffs, country.lower())
        # check if model is trained, else train model
        modelname = f"autoencoder_{country.lower()}_res{res}.h5"
        if os.path.exists(Path(model_dir) / modelname):
            print("--- Model already saved.")
        else:
            print("--- Training auto encoder...")
            gaf.train_auto_encoder(
                list(ctry.hex_code), tiffs, hyper_tuning, model_dir, country, res
            )
        # check if autoencoder features have been saved
        filename = f"encodings_{country.lower()}_res{res}.csv"
        if os.path.exists(Path(save_dir) / filename):
            print("--- Autoencoding features have already been saved.")
            auto_features = pd.read_csv(Path(save_dir) / filename)
        else:
            print("--- Retrieving autoencoding features...")
            auto_features = gaf.retrieve_autoencoder_features(
                list(ctry.hex_code), model_dir, country, res, tiffs, gpu
            )
            auto_features = (
                pd.DataFrame(
                    data=auto_features,
                    columns=["f_" + str(i) for i in range(auto_features.shape[1])],
                    index=list(ctry.hex_code),
                )
                .reset_index()
                .rename({"index": "hex_code"}, axis=1)
            )
            print(f"--- Saving autoencoding features to {save_dir}...")
            auto_features.to_csv(
                Path(save_dir) / filename,
                index=False,
            )
        hexes = hexes.merge(auto_features, on="hex_code", how="left")

    zero_fill_cols = [
        "n_conflicts",
        "GSM",
        "LTE",
        "NR",
        "UMTS",
    ]
    # where nans mean zero, fill as such
    hexes.fillna(value={col: 0 for col in zero_fill_cols}, inplace=True)
    logger.info("Finishing process...")
    return hexes


@g.timing
def create_dataset(
    country_code,
    country,
    res,
    gpu=False,
    encoders=True,
    force=False,
    force_download=False,
    audience=False,
    hyper_tuning=True,
    lat="latnum",
    long="longnum",
    interim_dir=c.int_data,
    save_dir=c.proc_data,
    model_dir=c.base_dir_model,
    threshold=c.cutoff,
    read_dir_target=c.raw_data,
    read_dir=c.ext_data,
    tiff_dir=c.tiff_data,
) -> pd.DataFrame:
    """Create dataset
    :param country_code: country code
    :type country_code: str
    :param country: country of interest
    :type country: str
    :param res: grid resolution
    :type res: int
    :param gpu: whether to use gpus or not
    :type gpu: bool
    :param encoders: whether to append autoencoder features
    :type encoders: bool
    :param force: force clipping, defaults to False
    :type force: bool, optional
    :param force_download: force download, defaults to False
    :type force_download: bool, optional
    :param audience: whether or not to include audience estimates, defaults to False
    :type audience: bool, optional
    :param hyper_tuning: whether or not to perform hyperparameter tuning, defaults to False
    :type hyper_tuning: bool, optional
    :param lat: colname containing latitude, defaults to "latnum"
    :type lat: str, optional
    :param long: colname containing longitude, defaults to "longnum"
    :type long: str, optional
    :param interim_dir: path to interim data, defaults to c.int_data
    :type interim_dir: str, optional
    :param read_dir: path to read input, defaults to c.ext_data
    :type read_dir: str, optional
    :param save_dir: path to save output, defaults to c.proc_data
    :type save_dir: str, optional
    :param model_dir: path to model, defaults to c.base_dir_model
    :type model_dir: str, optional
    :param tiff_dir: path to tiff files, defaults to c.tiff_data
    :type tiff_dir: str, optional
    :param threshold: minimum number of surveys per hexagon, defaults to c.cutoff
    :type threshold: int, optional
    :param read_dir_target: path to directory of target data, defaults to c.raw_data
    :type read_dir_target: str, optional
    :return: dataset with features and target variable
    :rtype: dataframe
    """
    tprint("Child Poverty Index", font="cybermedum")
    print(f"Building dataset for {country} at resolution {res}")
    print(
        f"Creating target variable...only available for certain hexagons in {country}"
    )
    train = create_target_variable(
        country_code, res, lat, long, threshold, read_dir_target
    )
    train_expanded = create_target_variable(
        country_code, res, lat, long, threshold, read_dir_target, copy_to_nbrs=True
    )
    print(
        f"Appending  features to all hexagons in {country}. This step might take a while...~10 minutes"
    )
    complete = append_features_to_hexes(
        country,
        res,
        encoders,
        gpu,
        force=force,
        force_download=force_download,
        audience=audience,
        read_dir=read_dir,
        save_dir=interim_dir,
        model_dir=model_dir,
        tiff_dir=tiff_dir,
        hyper_tuning=hyper_tuning,
    )
    print(f"Merging target variable to hexagons in {country}")
    complete = complete.merge(train, on="hex_code", how="left")
    print(f"Saving dataset to {save_dir}")
    complete.to_csv(
        Path(save_dir) / f"hexes_{country.lower()}_res{res}_thres{threshold}.csv",
        index=False,
    )
    print("complete:", len(complete))
    train_expanded.to_csv(
        Path(save_dir) / f"expanded_{country.lower()}_res{res}_thres{threshold}.csv",
        index=False,
    )
    print("Done!")
    return complete


if __name__ == "__main__":

    parser = argparse.ArgumentParser("High-res multi-dim CPI dataset creation")

    parser.add_argument(
        "-c",
        "--country",
        type=str,
        help="Country to make dataset for, default is Nigeria",
        default="Nigeria",
    )

    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        help="H3 resolution level, default is 7",
        default=7,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of dataset, without redownloading unless necessary",
    )
    parser.add_argument(
        "--force-download",
        "-fdl",
        action="store_true",
        help="Force (re)download of dataset",
    )
    parser.add_argument(
        "--add-auto", action="store_true", help="Generate autoencoder features also"
    )

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(0)

    country = pycountry.countries.search_fuzzy(args.country)[0]
    country_name = country.name
    country_code = country.alpha_3

    if args.add_auto:
        num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
        print("Num GPUs Available: ", num_gpus)
        if num_gpus > 0:
            gpu = True
        else:
            gpu = False

    else:
        gpu = False

    create_dataset(
        country_code=country_code,
        country=country_name,
        gpu=gpu,
        res=args.resolution,
        force=args.force,
        force_download=args.force_download,
        encoders=args.add_auto,
    )
