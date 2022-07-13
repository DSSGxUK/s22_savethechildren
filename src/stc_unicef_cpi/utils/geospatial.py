import geopandas as gpd
import h3.api.numpy_int as h3


def get_lat_long(df, geo_col):
    df["lat"] = df[geo_col].map(lambda p: p.x)
    df["long"] = df[geo_col].map(lambda p: p.y)
    return df


def create_geometry(df, lat, long):
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lat], df[long]))
    return df


def get_hex_code(df, lat, long):

    df["hex_code"] = df[[lat, long]].apply(
        lambda row: h3.geo_to_h3(row[lat], row[long], 7), axis=1
    )
    return df


def aggregate_hexagon(df, col_to_agg, name_agg, type):
    if type == "count":
        df = df.groupby("hex_code").count().reset_index()
    else:
        df = df.groupby("hex_code").mean().reset_index()
    df = df[["hex_code", col_to_agg]]
    df = df.rename({col_to_agg: name_agg}, axis=1)
    return df
