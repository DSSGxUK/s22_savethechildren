# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import os.path
import glob as glob
import rioxarray as rxr
import shapely.wkt
import numpy as np

import src.stc_unicef_cpi.utils.constants as c
import src.stc_unicef_cpi.data.process_geotiff as pg
import src.stc_unicef_cpi.utils.general as g
import src.stc_unicef_cpi.data.get_econ_data as econ
import src.stc_unicef_cpi.data.process_netcdf as net
import src.stc_unicef_cpi.data.get_cell_tower_data as cell
import src.stc_unicef_cpi.utils.geospatial as geo

#from src.stc_unicef_cpi.data.stream_data import RunStreamer

from functools import reduce, partial
from datetime import date


def read_input_unicef(path_read):
    """read_input_unicef _summary_

    _extended_summary_

    :param path_read: _description_
    :type path_read: _type_
    :return: _description_
    :rtype: _type_
    """
    df = pd.read_csv(path_read)
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

    _extended_summary_

    :param df: _description_
    :type df: _type_
    :return: _description_
    :rtype: _type_
    """

    df_mean = df.groupby(by=["hex_code"], as_index=False).mean()
    df_count = df.groupby(by=["hex_code"], as_index=False).count()[['hex_code', 'survey']]
    return df_mean, df_count


def create_target_variable(country_code, res, lat, long, threshold, read_dir):
    source = f"{read_dir}/childpoverty_microdata_gps_21jun22.csv"
    df = read_input_unicef(source)
    sub = select_country(df, country_code, lat, long)
    sub = geo.get_hex_code(sub, lat, long, res)
    sub = sub.reset_index(drop=True)
    sub_mean, sub_count = aggregate_dataset(sub)
    sub_count = sub_count[sub_count.survey>=threshold]
    survey = geo.get_hex_centroid(sub_mean, "hex_code")
    survey_threshold = sub_count.merge(survey, how='left', on='hex_code')
    return survey_threshold


def change_name_reproject_tiff(tiff, attributes, country, read_dir=c.ext_data):
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
        data.attrs["long_name"] = attributes
        data.rio.to_raster(tiff)
        p_r = f"{read_dir}/gee/cpi_poptotal_{country.lower()}_500.tif"
        pg.rxr_reproject_tiff_to_target(tiff, p_r, tiff, verbose=True)


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
    net.netcdf_to_clipped_array(f"{read_dir}/gdp_ppp_30.nc", ctry_name=country, save_dir=out_dir)

    # clip ec and gdp
    tifs = f"{read_dir}/*/*/2019/*.tif"
    partial_func = partial(pg.clip_tif_to_ctry, ctry_name=country, save_dir=out_dir)
    list(map(partial_func, glob.glob(tifs)))

    # reproject resolution + crs
    econ_tiffs = sorted(glob.glob(f"{out_dir}/{country.lower()}*.tif"))
    attributes = [["GDP_2019"], ["EC_2019"], ["GDP_PPP_1990", "GDP_PPP_2000", "GDP_PPP_2015"]]
    mapfunc = partial(change_name_reproject_tiff, country=country)
    list(map(mapfunc, econ_tiffs, attributes))

    # critical infrastructure data
    cisi = glob.glob(f"{read_dir}/*/*/010_degree/global.tif")
    partial_func = partial(pg.clip_tif_to_ctry, ctry_name=country, save_dir=out_dir)
    list(map(partial_func, cisi))
    p_r = f"{read_dir}/gee/cpi_poptotal_{country.lower()}_500.tif"
    pg.rxr_reproject_tiff_to_target(cisi[0], p_r, cisi[0], verbose=True)


def preprocessed_speed_test(speed, res, country):
    speed["geometry"] = speed.geometry.swifter.apply(shapely.wkt.loads)
    speed = gpd.GeoDataFrame(speed, crs="epsg:4326")
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    speed = gpd.sjoin(speed, world[world.name == country], how="inner", op="intersects").reset_index(drop=True)
    tmp = speed.geometry.swifter.apply(lambda x: pd.Series(np.array(x.centroid.coords.xy).flatten()))
    speed[["long", "lat"]] = tmp
    speed = geo.get_hex_code(speed, "lat", "long", res)
    speed = speed[["hex_code", "avg_d_kbps", "avg_u_kbps"]].groupby("hex_code").mean().reset_index()
    return speed


def preprocessed_commuting_zones(country, res, read_dir=c.ext_data):
    """Preprocess commuting zones"""
    commuting = pd.read_csv(f"{read_dir}/commuting_zones.csv")
    commuting = commuting[commuting['country'] == country]
    comm = list(commuting['geometry'])
    comm_zones = pd.concat(list(map(partial(geo.hexes_poly, res=res), comm)))
    comm_zones = comm_zones.merge(commuting, on='geometry', how='left')
    comm_zones = comm_zones.add_suffix('_commuting')
    comm_zones.rename(columns={'hex_code_commuting': 'hex_code'}, inplace=True)

    return comm_zones


def append_features_to_hexes(
    country, res, read_dir=c.ext_data, save_dir=c.int_data
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
    # TODO: Integrate satellite information to pipeline
    # Country hexes
    hexes_ctry = geo.get_hexes_for_ctry(country, res)
    ctry = pd.DataFrame(hexes_ctry, columns=['hex_code'])
    
    # Retrieve external data
    # RunStreamer(country, force=forced_download)

    # Facebook connectivity metrics
    #connect_fb = pd.read_parquet(f'fb_aud_{country.lower()}_res{res}.parquet')
    #connect_fb = geo.get_hex_centroid(connect_fb)

    # Conflict Zones
    cz = pd.read_csv(f"{read_dir}/conflict/GEDEvent_v22_1.csv")
    cz = cz[cz.country == country]
    cz = geo.create_geometry(cz, "latitude", "longitude")
    cz = geo.get_hex_code(cz, "latitude", "longitude", res)
    cz = geo.aggregate_hexagon(cz, "geometry", "n_conflicts", "count")

    # Commuting zones
    commuting = preprocessed_commuting_zones(country, res, read_dir)[c.cols_commuting]

    # Critical Infrastructure
    file_cisi = f"{save_dir}/{country.lower()}_global.tif"
    cis = pg.geotiff_to_df(file_cisi)
    cis = geo.create_geometry(cis, "latitude", "longitude")
    cis = geo.get_hex_code(cis, "latitude", "longitude", res)
    cis = geo.aggregate_hexagon(cis, f"{country.lower()}_global", "csi", "mean")
    # new = pg.agg_tif_to_df(ctry, file_cisi, rm_prefix=f'{country.lower()}_', verbose=True)

    # Road density
    road = pd.read_csv(f"{read_dir}/road_density_{country.lower()}_res{res}.csv")

    # Speed Test
    speed = pd.read_csv(f'{read_dir}/2021-10-01_performance_mobile_tiles.csv')
    speed = preprocessed_speed_test(speed, res, country)

    ## Open Cell Data
    #cell = get_cell_data(country)
    #cell = create_geometry(cell, "lat", "long")
    #cell = get_hex_code(cell, "lat", "long")
    #cell = aggregate_hexagon(cell, "cid", "cells", "count")

    # Aggregate Data
    #dfs = [ctry, commuting, cz, cis, road, speed]#, connect_fb]#, cell]
    #sub = reduce(
    #    lambda left, right: pd.merge(left, right, on="hex_code", how="left"), dfs
    #)

    #return sub


append_features_to_hexes(
    country='Senegal', res=6, read_dir=c.ext_data, save_dir=c.int_data
)


def append_target_variable_to_hexes(
    country_code,
    country,
    res,
    lat="latnum",
    long="longnum",
    save_dir=c.int_data,
    threshold=c.cutoff,
    read_dir=c.raw_data
):
    train = create_target_variable(country_code, res, lat, long, threshold, read_dir)
    complete = append_features_to_hexes(country, read_dir, save_dir)
    complete = complete.merge(train, on='hex_code', how='left')
    return complete

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
