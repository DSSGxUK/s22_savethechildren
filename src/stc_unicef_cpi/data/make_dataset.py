# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import h3.api.numpy_int as h3

# Health Sites
df = pd.read_csv("nga_health.csv")
# Remove health facilities without coordinates
df = df[~df.X.isna()]
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y))
df["hex_code"] = df[["X", "Y"]].apply(
    lambda row: h3.geo_to_h3(row["X"], row["Y"], 7), axis=1
)
print(df.groupby("hex_code")["geometry"].count())

# Education Facilities
df = gpd.read_file("nga_education")
print(df.columns)
print(df)
df["lat"] = df.geometry.map(lambda p: p.x)
df["long"] = df.geometry.map(lambda p: p.y)
df["hex_code"] = df[["lat", "long"]].apply(
    lambda row: h3.geo_to_h3(row["lat"], row["long"], 7), axis=1
)
print(df)

# Critical Infrastructure


# Conflict Zones
df = pd.read_csv("conflict/GEDEvent_v22_1.csv")
df = df[df.country == "Nigeria"]
df["latitude"] = df.geometry.map(lambda p: p.x)
df["longitude"] = df.geometry.map(lambda p: p.y)
df["hex_code"] = df[["latitude", "longitude"]].apply(
    lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], 7), axis=1
)
print(df.columns)
