# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd

from functools import reduce

from src.stc_unicef_cpi.data.process_geotiff import geotiff_to_df
from src.stc_unicef_cpi.data.get_facebook_data import get_facebook_estimates

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


def append_predictor_variables(country_code="NGA", lat="latnum", long="longnum", res=6):
    sub = create_target_variable(country_code, lat, long, res)
    sub = sub.head(10)
    name_out = "fb_nigeria_train.parquet"
    coords = sub["hex_centroid"].values
    # Facebook connectity metrics
    connect_fb = get_facebook_estimates(coords, name_out, res)
    sub = sub.merge(connect_fb, on="hex_centroid", how="left")
    print(sub)


append_predictor_variables()

# def get_
# osm_nga.hex_id = osm_nga.hex_id.swifter.apply(h3.string_to_h3)
# osm_nga.rename(columns={"hex_id": "hex_code"}, inplace=True)
# osm_nga.drop(columns=["geometry"], inplace=True)
# nga_df = nga_df.merge(
#    osm_nga.groupby(by=["hex_code"], as_index=False).mean(), how="left", on="hex_code"
# )
# sub = create_target_variable()
# print(sub)

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
#
## Critical Infrastructure
#
# ci = geotiff_to_df(
#    "../../stc_unicef_cpi/data/infrastructure/CISI/010_degree/africa.tif"
# )
# ci = create_geometry(ci, "latitude", "longitude")
# ci = get_hex_code(ci, "latitude", "longitude")
# ci = aggregate_hexagon(ci, "fric", "cii", "mean")
#
## Conflict Zones
# cz = pd.read_csv("conflict/GEDEvent_v22_1.csv")
# cz = cz[cz.country == "Nigeria"]
# cz = create_geometry(cz, "latitude", "longitude")
# cz = get_hex_code(cz, "latitude", "longitude")
# cz = aggregate_hexagon(cz, "geometry", "n_conflicts", "count")
#
## Merge into original dataset
# name_file = "nga_clean_v1.csv"
# df = pd.read_csv(name_file)
# dfs = [df, hh, edu, ci, cz]
# df = reduce(lambda left, right: pd.merge(left, right, on="hex_code", how="left"), dfs)
# df.to_csv(name_file, index=False)
