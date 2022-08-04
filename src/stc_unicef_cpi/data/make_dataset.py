# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import os.path
import glob as glob
import rioxarray as rxr
import h3.api.numpy_int as h3
import shapely.wkt

from shapely import geometry
import src.stc_unicef_cpi.utils.constants as c
import src.stc_unicef_cpi.data.process_geotiff as pg
import src.stc_unicef_cpi.utils.general as g
import src.stc_unicef_cpi.data.get_econ_data as econ
import src.stc_unicef_cpi.data.process_netcdf as net
import src.stc_unicef_cpi.data.get_facebook_data as fb
import src.stc_unicef_cpi.data.get_osm_data as osm
import src.stc_unicef_cpi.data.get_cell_tower_data as cell
import src.stc_unicef_cpi.data.get_satellite_data as ge
import src.stc_unicef_cpi.utils.geospatial as geo
import src.stc_unicef_cpi.data.get_speedtest_data as speed
#import src.stc_unicef_cpi.data.stream_data as stream

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

    df = df.groupby(by=["hex_code"], as_index=False).mean()

    return df


def create_target_variable(country_code, lat, long, res, read_dir=c.raw_data):
    source = f"{read_dir}/childpoverty_microdata_gps_21jun22.csv"
    df = read_input_unicef(source)
    sub = select_country(df, country_code, lat, long)
    sub = geo.get_hex_code(sub, lat, long, res)
    sub = sub.reset_index(drop=True)
    sub = aggregate_dataset(sub)
    sub = geo.get_hex_centroid(sub, "hex_code")

    return sub


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


def commuting_zone(point, resolution):
    geom = shapely.wkt.loads(point)
    if geom.geom_type == 'MultiPolygon':
        polygons = list(geom.geoms)
    elif geom.geom_type == 'Polygon':
        polygons = [geom]
    else:
        raise IOError('Shape is not a polygon.')
    hexs = [h3.polyfill(geometry.mapping(polygon), resolution, geo_json_conformant=True) for polygon in polygons]
    hexs = list(set([item for sublist in hexs for item in sublist]))
    df = pd.DataFrame(hexs)
    df.rename(columns={0:'hex_code'}, inplace=True)
    df['geometry'] = point
    return df


def preprocessed_commuting_zones(country, res, read_dir=c.ext_data):
    commuting = pd.read_csv(f"{read_dir}/commuting_zones.csv")
    commuting = commuting[commuting['country'] == country]
    comm = list(commuting['geometry'])
    comm_zones = pd.concat(list(map(partial(commuting_zone, resolution=res), comm)))
    comm_zones = comm_zones.merge(commuting, on='geometry', how='left')

    return comm_zones


def append_predictor_variables(
    country_code="NGA", country="Nigeria", lat="latnum", long="longnum", res=6, forced_download=True
):
    """append_predictor_variables _summary_

    _extended_summary_

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
    # TODO: Include threshold to pipeline
    sub = create_target_variable(country_code, lat, long, res)
    # countries hexes
    hexes = get_hexes_for_ctry(country, res)
    ctry = pd.DataFrame(hexes, columns=['hex_code'])
    ctry = get_hex_centroid(ctry, "hex_code")
    ctry_name = country_code.lower()
    today = date.today()
    dat_scp = today.strftime("%d-%m-%Y")
    name_out = f"fb_{ctry_name}_res{res}_{dat_scp}.parquet"

    stream.RunStreamer(country, force=forced_download)

    ## Facebook connectivity metrics
    connect_fb = get_facebook_estimates(ctry["hex_centroid"].values[0:900], name_out, res)
    #sub = sub.merge(connect_fb, on=["hex_centroid", "lat", "long"], how="left")
#

    ## Critical Infrastructure
    #ci = geotiff_to_df(f"{path_data}infrastructure/CISI/010_degree/global.tif")
    #ci = create_geometry(ci, "latitude", "longitude")
    #ci = get_hex_code(ci, "latitude", "longitude")
    #ci = aggregate_hexagon(ci, "fric", "cii", "mean")
#
    ## Conflict Zones
    #cz = pd.read_csv(f"{path_data}conflict/GEDEvent_v22_1.csv")
    #cz = cz[cz.country == country]
    #cz = create_geometry(cz, "latitude", "longitude")
    #cz = get_hex_code(cz, "latitude", "longitude")
    #cz = aggregate_hexagon(cz, "geometry", "n_conflicts", "count")
#
    ## Open Cell Data
    #cell = get_cell_data(country)
    #cell = create_geometry(cell, "lat", "long")
    #cell = get_hex_code(cell, "lat", "long")
    #cell = aggregate_hexagon(cell, "cid", "cells", "count")
#
    ## Road density
    #road = get_road_density(country, res)

    # Speet Test
    url, name = get_speedtest_url(service_type='mobile', year=2021, q=4)
    file_exists = os.path.exists(f"{path_data}connectivity/GEDEvent_v22_1.csv")

    get_speedtest_info(url, name)

    
    ## Aggregate Data
    #dfs = [sub, cell, ci, cz, road]
    #sub = reduce(
    #    lambda left, right: pd.merge(left, right, on="hex_code", how="left"), dfs
    #)



#append_predictor_variables()


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
