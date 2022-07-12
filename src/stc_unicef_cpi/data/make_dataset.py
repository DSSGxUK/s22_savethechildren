# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import geopandas as gpd

# Health Sites
df = pd.read_csv("nga_health.csv")
# Remove health facilities without coordinates
df = df[~df.X.isna()]
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y))
print(df)

# Education Facilities
df = gpd.read_file("nga_education")
print(df.columns)


# Critical Infrastructure


# Conflict Zones
