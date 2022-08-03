# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import os.path
import glob as glob

from functools import reduce, partial
from datetime import date

import src.stc_unicef_cpi.utils.constants as c
import src.stc_unicef_cpi.data.process_geotiff as pg
import src.stc_unicef_cpi.utils.general as g
import src.stc_unicef_cpi.data.get_econ_data as econ
import src.stc_unicef_cpi.data.process_netcdf as net

from src.stc_unicef_cpi.data.get_facebook_data import get_facebook_estimates
from src.stc_unicef_cpi.data.get_osm_data import get_road_density
from src.stc_unicef_cpi.data.get_cell_tower_data import get_cell_data
#from src.stc_unicef_cpi.data.get_satellite_data import SatelliteImages
from src.stc_unicef_cpi.utils.geospatial import (
    create_geometry,
    get_hex_code,
    get_hex_centroid,
    get_hexes_for_ctry,
    aggregate_hexagon,
    get_lat_long,
)

from src.stc_unicef_cpi.data.get_speedtest_data import get_speedtest_url, get_speedtest_info


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


def preprocessed_tif_files(country, out_dir=c.int_data):

    g.create_folder(out_dir)
    # clip gdp ppp 30 arc sec
    net.netcdf_to_clipped_array(
        f"{c.ext_data}/gdp_ppp_30.nc", ctry_name=country, save_dir=out_dir
        )
    # clip ec and gdp
    tifs = f"{c.ext_data}/*/*/2019/*.tif"
    partial_func = partial(pg.clip_tif_to_ctry, ctry_name=country, save_dir=out_dir)
    list(map(partial_func, glob.glob(tifs)))

    # reproject resolution + crs
    #econ_tiffs = glob.glob(str(econ_dir / "*.tif"))
    #econ_tiffs
    #for i, econ_tiff in enumerate(econ_tiffs):
    #    with rxr.open_rasterio(econ_tiff) as data:
    #        name = Path(econ_tiff).name
    #        if "GDP_PPP" in name:
    #            data.attrs["long_name"] = ["GDP_PPP_1990", "GDP_PPP_2000", "GDP_PPP_2015"]
    #        elif "2019GDP" in name:
    #            data.attrs["long_name"] = ["GDP_2019"]
    #        elif "EC" in name:
    #            data.attrs["long_name"] = ["EC_2019"]
    #        data.rio.to_raster(econ_tiff)
    #    pg.rxr_reproject_tiff_to_target(
    #        econ_tiff,
    #        glob.glob(str(tiff_dir / "*.tif"))[0],
    #        tiff_dir / Path(econ_tiff).name,
    #        verbose=True
    #    )

preprocessed_tif_files(country='Senegal')

def append_predictor_variables(
    country_code="NGA", country="Nigeria", lat="latnum", long="longnum", res=6
):
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

    ## Facebook connectivity metrics
    connect_fb = get_facebook_estimates(ctry["hex_centroid"].values[0:900], name_out, res)
    #sub = sub.merge(connect_fb, on=["hex_centroid", "lat", "long"], how="left")
#
    ## Download data if it does not exist
    file_exists = os.path.exists(f"{path_data}conflict/GEDEvent_v22_1.csv")
    if file_exists:
        pass
    else:
        download_econ_data()

    ## Critical Infrastructure
    #ci = geotiff_to_df(f"{path_data}infrastructure/CISI/010_degree/global.tif")
    clip_tif_to_ctry
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
    if file_exists:
        pass
    else:
        download_econ_data(path_data)
    get_speedtest_info(url, name)

    # Satellite Data

    SatelliteImages(country).get_satellite_images()

    
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
