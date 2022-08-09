import argparse
import glob as glob
import sys
import logging
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import shapely.wkt

import stc_unicef_cpi.data.process_geotiff as pg
import stc_unicef_cpi.data.process_netcdf as net
import stc_unicef_cpi.utils.constants as c
import stc_unicef_cpi.utils.general as g
import stc_unicef_cpi.utils.geospatial as geo

from stc_unicef_cpi.data.stream_data import RunStreamer
from pathlib import Path
from functools import partial, reduce


def read_input_unicef(path_read):
    """read_input_unicef _summary_
    :param path_read: _description_
    :type path_read: _type_
    :return: _description_
    :rtype: _type_
    """
    df = pd.read_csv(path_read, low_memory=False)
    return df


def select_country(df, country_code, lat, long):
    """Select country of interest
    :param df: _description_
    :type df: _type_
    :param country_code: _description_
    :type country_code: _type_
    :param lat: _description_
    :type lat: _type_
    :param long: _description_
    :type long: _type_
    :return: _description_
    :rtype: _type_
    """
    df.columns = df.columns.str.lower()
    subset = df[df["countrycode"].str.strip() == country_code]
    subset.dropna(subset=[lat, long], inplace=True)
    return subset


def aggregate_dataset(df):
    """aggregate_dataset _summary_
    :param df: _description_
    :type df: _type_
    :return: _description_
    :rtype: _type_
    """
    df_mean = df.groupby(by=["hex_code"], as_index=False).mean()
    df_count = df.groupby(by=["hex_code"], as_index=False).count()[
        ["hex_code", "survey"]
    ]
    return df_mean, df_count


def create_target_variable(country_code, res, lat, long, threshold, read_dir):
    source = Path(read_dir) / "childpoverty_microdata_gps_21jun22.csv"
    df = read_input_unicef(source)
    sub = select_country(df, country_code, lat, long)
    sub = geo.get_hex_code(sub, lat, long, res)
    sub = sub.reset_index(drop=True)
    sub_mean, sub_count = aggregate_dataset(sub)
    sub_count = sub_count[sub_count.survey >= threshold]
    survey = geo.get_hex_centroid(sub_mean, "hex_code")
    survey_threshold = sub_count.merge(survey, how="left", on="hex_code")
    return survey_threshold


def change_name_reproject_tiff(tiff, attribute, country, read_dir=c.ext_data, out_dir=c.int_data):
    """Rename attributes and reproject Tiff file
    :param tiff: _description_
    :type tiff: _type_
    :param attributes: _description_
    :type attributes: _type_
    :param country: _description_
    :type country: _type_
    :param read_dir: _description_, defaults to c.ext_data
    :type read_dir: _type_, optional
    """
    with rxr.open_rasterio(tiff) as data:
        fname = Path(tiff).name
        data.attrs["long_name"] = attribute
        data.rio.to_raster(tiff)
        p_r = Path(read_dir) / "gee" / f"cpi_poptotal_{country.lower()}_500.tif"
        pg.rxr_reproject_tiff_to_target(tiff, p_r, Path(out_dir) / fname, verbose=True)


@g.timing
def preprocessed_tiff_files(country, read_dir=c.ext_data, out_dir=c.int_data):
    """Preprocess tiff files
    :param country: _description_
    :type country: _type_
    :param read_dir: _description_, defaults to c.ext_data
    :type read_dir: _type_, optional
    :param out_dir: _description_, defaults to c.int_data
    :type out_dir: _type_, optional
    """
    g.create_folder(out_dir)
    # clip gdp ppp 30 arc sec
    print("Clipping gdp pp 30 arc sec")
    net.netcdf_to_clipped_array(
        Path(read_dir) / "gdp_ppp_30.nc", ctry_name=country, save_dir=read_dir
    )

    # clip ec and gdp
    print("Clipping ec and gdp")
    tifs = glob.glob(str(Path(read_dir) / "*" / "*" / "2019" / "*.tif"))
    partial_func = partial(pg.clip_tif_to_ctry, ctry_name=country, save_dir=read_dir)
    list(map(partial_func, tifs))

    # reproject resolution + crs
    print("Reprojecting resolution & determining crs")
    econ_tiffs = sorted(glob.glob(str(Path(read_dir) / f"{country.lower()}_*.tif")))
    econ_tiffs = [ele for ele in econ_tiffs if "africa" not in ele]
    attributes = [
        ["gdp_2019"],
        ["ec_2019"],
        ["gdp_ppp_1990", "gdp_ppp_2000", "gdp_ppp_2015"]
    ]
    mapfunc = partial(change_name_reproject_tiff, country=country)
    list(map(mapfunc, econ_tiffs, attributes))

    # critical infrastructure data
    print("Reprojecting critical infrastructure data")
    cisi = glob.glob(str(Path(read_dir) / "*" / "*" / "010_degree" / "africa.tif"))[0]
    pg.clip_tif_to_ctry(cisi, ctry_name=country, save_dir=read_dir)
    p_r = Path(read_dir) / "gee" / f"cpi_poptotal_{country.lower()}_500.tif"
    cisi_ctry = Path(read_dir) / f"{country.lower()}_africa.tif"
    fname = Path(cisi_ctry).name
    pg.rxr_reproject_tiff_to_target(cisi_ctry, p_r, Path(out_dir) / fname, verbose=True)


@g.timing
def preprocessed_speed_test(speed, res, country):
    speed["geometry"] = speed.geometry.swifter.apply(shapely.wkt.loads)
    speed = gpd.GeoDataFrame(speed, crs="epsg:4326")
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    speed = gpd.sjoin(
        speed, world[world.name == country], how="inner", op="intersects"
    ).reset_index(drop=True)
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
def preprocessed_commuting_zones(country, res, read_dir=c.ext_data):
    """Preprocess commuting zones"""
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
    country, res, force=False, audience=False, read_dir=c.ext_data, save_dir=c.int_data
):
    """Append features to hexagons withing a country
    :param country_code: _description_, defaults to "NGA"
    :type country_code: str, optional
    :param country: _description_, defaults to "Nigeria"
    :type country: str, optional
    :param lat: _description_, defaults to "latnum"
    :type lat: str, optional
    :param long: _description_, defaults to "longnum"
    :type long: str, optional
    :param res: _description_, defaults to 6
    :type res: int, optional
    """
    # Setting up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    file_handler = logging.FileHandler('make_dataset.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('Starting process...')

    # Country hexes
    logger.info(
        f'Retrieving hexagons for {country} at resolution {res}.'
        )
    hexes_ctry = geo.get_hexes_for_ctry(country, res)
    ctry = pd.DataFrame(hexes_ctry, columns=["hex_code"])

    # Retrieve external data
    logger.info(
        f"Initiating data retrieval. Audience: {audience}. Forced data gathering: {force}"
    )
    RunStreamer(country, res, force, audience)
    logger.info('Finished data retrieval.')
    logger.info(
        f"Please check your 'gee' folder in google drive and download all content to {read_dir}/gee."
    )
    time.sleep(100)

    # Facebook connectivity metrics
    if audience:
        logger.info(f'Collecting audience estimates for {country} at resolution {res}...')
        fb = pd.read_parquet(Path(read_dir) / f"fb_aud_{country.lower()}_res{res}.parquet")
        fb = geo.get_hex_centroid(fb)

    # Preprocessed tiff files
    logger.info(f'Preprocessing tiff files from {read_dir} and saving to {save_dir}..')
    preprocessed_tiff_files(country, read_dir, save_dir)

    # Conflict Zones
    logger.info("Reading and computing conflict zone estimates...")
    cz = pd.read_csv(Path(read_dir) / "conflict/GEDEvent_v22_1.csv")
    cz = cz[cz.country == country]
    cz = geo.create_geometry(cz, "latitude", "longitude")
    cz = geo.get_hex_code(cz, "latitude", "longitude", res)
    cz = geo.aggregate_hexagon(cz, "geometry", "n_conflicts", "count")

    # Commuting zones
    logger.info("Reading and computing commuting zone estimates...")
    commuting = preprocessed_commuting_zones(country, res, read_dir)[c.cols_commuting]

    # Economic data
    logger.info("Retrieving features from economic tif files...")
    econ_files = glob.glob(str(Path(save_dir) / f"{country.lower()}*.tif"))
    econ_files = [ele for ele in econ_files if "ppp" not in ele]
    econ = list(map(pg.geotiff_to_df, econ_files))
    econ = reduce(
        lambda left, right: pd.merge(left, right, on=["latitude", "longitude"], how="outer"), econ
    )
    ppp = glob.glob(str(Path(save_dir) / f"{country.lower()}*ppp*.tif"))[0]
    ppp = pg.geotiff_to_df(ppp, ["gdp_ppp_1990", "gdp_ppp_2000", "gdp_ppp_2015"])
    econ = econ.merge(ppp, on=["latitude", "longitude"], how="outer")

    # Google Earth Engine
    logger.info("Retrieving features from google earth engine tif files...")
    gee_files = glob.glob(str(Path(read_dir) / "gee" / f"*_{country.lower()}*.tif"))
    gee = list(map(pg.geotiff_to_df, gee_files))
    gee = reduce(
        lambda left, right: pd.merge(left, right, on=["latitude", "longitude"], how="outer"), gee
    )

    # Join GEE with Econ
    logger.info("Merging and aggregating features from tiff files to hexagons...")
    images = gee.merge(econ, on=["latitude", "longitude"], how="outer")
    images = geo.create_geometry(images, "latitude", "longitude")
    images = geo.get_hex_code(images, "latitude", "longitude", res)
    images = images.groupby("hex_code").mean().reset_index()
    images = images.drop(["latitude", "longitude"], axis=1)

    # Road density
    logger.info("Reading road density estimates...")
    road = pd.read_csv(Path(read_dir) / f"road_density_{country.lower()}_res{res}.csv")

    # Speed Test
    logger.info("Reading speed test estimates...")
    speed = pd.read_csv(Path(read_dir) / "2021-10-01_performance_mobile_tiles.csv")
    speed = preprocessed_speed_test(speed, res, country)

    # Open Cell Data
    logger.info("Reading open cell data...")
    cell = g.read_csv_gzip(glob.glob(str(Path(read_dir) / f"{country.lower()}_*gz.tmp"))[0])
    cell = geo.create_geometry(cell, "lat", "long")
    cell = geo.get_hex_code(cell, "lat", "long", res)
    cell = geo.aggregate_hexagon(cell, "cid", "cells", "count")

    # Aggregate Data
    logger.info("Merging and aggregating features all features")
    dfs = [ctry, commuting, cz, road, speed, cell, images]
    hexes = reduce(
        lambda left, right: pd.merge(left, right, on="hex_code", how="left"), dfs
    )
    logger.info('Finishing process...')

    return hexes


@g.timing
def append_target_variable_to_hexes(
    country_code,
    country,
    res,
    force=False,
    audience=False,
    lat="latnum",
    long="longnum",
    save_dir=c.int_data,
    threshold=c.cutoff,
    read_dir_target=c.raw_data,
    read_dir=c.ext_data,

):
    print(f"Creating target variable...only available for certain hexagons in {country}")
    train = create_target_variable(country_code, res, lat, long, threshold, read_dir_target)
    print(f"Appending  features to all hexagons in {country}. This step might take a while...~20 minutes")
    complete = append_features_to_hexes(country, res, force, audience, read_dir, save_dir)
    print(f"Merging target variable to hexagons in {country}")
    complete = complete.merge(train, on="hex_code", how="left")
    print(f"Saving dataset to {save_dir}")
    complete.to_csv(Path(save_dir) / f"hexes_{country.lower()}_res{res}_thres{threshold}.csv", index=False)
    print("Done!")
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser("High-res multi-dim CPI model training")
    parser.add_argument(
        "-cc",
        "--country_code",
        type=str,
        help="Country code to make dataset for, default is NGA",
        default="NGA",
    )
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
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(0)
    append_target_variable_to_hexes(
        country_code=args.country_code,
        country=args.country,
        res=args.resolution
    )


## Health Sites
# hh = pd.read_csv("nga_health.csv")
# hh = hh[~hh.X.isna()]
# hh = create_geometry(hh, "X", "Y")
# hh = get_hex_code(hh, "X", "Y")
# hh = aggregate_hexagon(hh, "geometry", "n_health", "count")
#
#
## Education Facilities
# edu = gpd.read_file("nga_education")
# edu = get_lat_long(edu, "geometry")
# edu = get_hex_code(edu, "lat", "long")
# edu = aggregate_hexagon(edu, "geometry", "n_education", "count")
