# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import h3.api.numpy_int as h3

# Read

ori = pd.read_csv("../../stc_unicef_cpi/data/nga_clean_v1.csv")
print(ori.dtypes)

# Health Sites
# df = pd.read_csv("nga_health.csv")

# Remove health facilities without coordinates
# df = df[~df.X.isna()]
# df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y))
# df["hex_code"] = df[["X", "Y"]].apply(
#    lambda row: h3.geo_to_h3(row["X"], row["Y"], 7), axis=1
# )
# df = df.groupby("hex_code").count().reset_index()
# df = df[["hex_code", "geometry"]]
# df = df.rename({"geometry": "n_health"}, axis=1)
# ori = ori.merge(df, how="left", on="hex_code")
# ori.to_excel("nga_clean_v2.xlsx")

# Education Facilities
# df = gpd.read_file("nga_education")
# df["lat"] = df.geometry.map(lambda p: p.x)
# df["long"] = df.geometry.map(lambda p: p.y)
# df["hex_code"] = df[["lat", "long"]].apply(
#    lambda row: h3.geo_to_h3(row["lat"], row["long"], 7), axis=1
# )
# df = df.groupby("hex_code").count().reset_index()
# df = df[["hex_code", "geometry"]]
# df = df.rename({"geometry": "n_education"}, axis=1)
# ori = ori.merge(df, how="left", on="hex_code")
# ori.to_excel("nga_clean_v3.xlsx")


# Critical Infrastructure

# pd.read_tif("infrastructure/CISI/010_degree/africa.tif")

# Conflict Zones
df = pd.read_csv("conflict/GEDEvent_v22_1.csv")
df = df[df.country == "Nigeria"]
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.latitude, df.longitude))
df["hex_code"] = df[["latitude", "longitude"]].apply(
    lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], 7), axis=1
)
df = df.groupby("hex_code").count().reset_index()
df = df[["hex_code", "geometry"]]
df = df.rename({"geometry": "n_conflicts"}, axis=1)
ori = ori.merge(df, how="left", on="hex_code")
ori.to_csv("nga_clean_v1.csv")
