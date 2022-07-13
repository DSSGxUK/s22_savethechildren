# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd

from functools import reduce

from src.stc_unicef_cpi.data.process_geotiff import geotiff_to_df
from src.stc_unicef_cpi.utils.geospatial import (
    create_geometry,
    get_hex_code,
    aggregate_hexagon,
    get_lat_long,
)

# Health Sites
hh = pd.read_csv("nga_health.csv")
hh = hh[~hh.X.isna()]
hh = create_geometry(hh, "X", "Y")
hh = get_hex_code(hh, "X", "Y")
hh = aggregate_hexagon(hh, "geometry", "n_health", "count")


# Education Facilities
edu = gpd.read_file("nga_education")
edu = get_lat_long(edu, "geometry")
edu = get_hex_code(edu, "lat", "long")
edu = aggregate_hexagon(edu, "geometry", "n_education", "count")

# Critical Infrastructure

ci = geotiff_to_df(
    "../../stc_unicef_cpi/data/infrastructure/CISI/010_degree/africa.tif"
)
ci = create_geometry(ci, "latitude", "longitude")
ci = get_hex_code(ci, "latitude", "longitude")
ci = aggregate_hexagon(ci, "fric", "cii", "mean")

# Conflict Zones
cz = pd.read_csv("conflict/GEDEvent_v22_1.csv")
cz = cz[cz.country == "Nigeria"]
cz = create_geometry(cz, "latitude", "longitude")
cz = get_hex_code(cz, "latitude", "longitude")
cz = aggregate_hexagon(cz, "geometry", "n_conflicts", "count")

# Merge into original dataset
name_file = "nga_clean_v1.csv"
df = pd.read_csv(name_file)
dfs = [df, hh, edu, ci, cz]
df = reduce(lambda left, right: pd.merge(left, right, on="hex_code", how="left"), dfs)
df.to_csv(name_file, index=False)
