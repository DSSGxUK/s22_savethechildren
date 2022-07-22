# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import os.path

from functools import reduce

from src.stc_unicef_cpi.data.process_geotiff import geotiff_to_df
from src.stc_unicef_cpi.data.get_facebook_data import get_facebook_estimates
from src.stc_unicef_cpi.data.get_osm_data import get_road_density
from src.stc_unicef_cpi.data.get_econ_data import download_econ_data
from src.stc_unicef_cpi.data.get_cell_tower_data import get_cell_data
from src.stc_unicef_cpi.utils.geospatial import (
    create_geometry,
    get_hex_code,
    get_hex_centroid,
    aggregate_hexagon,
    get_lat_long,
)


def read_input_unicef(path_read):
    df = pd.read_csv(path_read)
    return df


def select_country(df, country_code, lat, long):
    df.columns = df.columns.str.lower()
    subset = df[df["countrycode"].str.strip() == country_code]
    subset.dropna(subset=[lat, long], inplace=True)
    return subset


def aggregate_dataset(df):

    df = df.groupby(by=["hex_code"], as_index=False).mean()

    return df


def create_target_variable(country_code, lat, long, res):
    source = "../../../data/childpoverty_microdata_gps_21jun22.csv"
    df = read_input_unicef(source)
    sub = select_country(df, country_code, lat, long)
    sub = get_hex_code(sub, lat, long, res)
    sub = sub.reset_index(drop=True)
    sub = aggregate_dataset(sub)
    sub = get_hex_centroid(sub, "hex_code")

    return sub


def append_predictor_variables(
    country_code="NGA", country="Nigeria", lat="latnum", long="longnum", res=6
):
    # TODO: Integrate satellite information to pipeline
    # TODO: Include threshold to pipeline
    sub = create_target_variable(country_code, lat, long, res)
    name_out = "fb_train.parquet"

    # Facebook connectivity metrics
    connect_fb = get_facebook_estimates(sub["hex_centroid"].values, name_out, res)
    sub = sub.merge(connect_fb, on=["hex_centroid", "lat", "long"], how="left")

    # Download data if it does not exist
    path_data = "../../../data/"
    file_exists = os.path.exists(f"{path_data}conflict/GEDEvent_v22_1.csv")
    if file_exists:
        pass
    else:
        download_econ_data(path_data)

    # Critical Infrastructure
    ci = geotiff_to_df(f"{path_data}infrastructure/CISI/010_degree/global.tif")
    ci = create_geometry(ci, "latitude", "longitude")
    ci = get_hex_code(ci, "latitude", "longitude")
    ci = aggregate_hexagon(ci, "fric", "cii", "mean")

    # Conflict Zones
    cz = pd.read_csv(f"{path_data}conflict/GEDEvent_v22_1.csv")
    cz = cz[cz.country == country]
    cz = create_geometry(cz, "latitude", "longitude")
    cz = get_hex_code(cz, "latitude", "longitude")
    cz = aggregate_hexagon(cz, "geometry", "n_conflicts", "count")

    # Open Cell Data
    cell = get_cell_data(country)
    cell = create_geometry(cell, "lat", "long")
    cell = get_hex_code(cell, "lat", "long")
    cell = aggregate_hexagon(cell, "cid", "cells", "count")

    # Road density
    road = get_road_density(country, res)

    # Aggregate Data
    dfs = [sub, cell, ci, cz, road]
    sub = reduce(
        lambda left, right: pd.merge(left, right, on="hex_code", how="left"), dfs
    )


append_predictor_variables()


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
