import geopandas as gpd
import h3.api.numpy_int as h3


def get_lat_long(df, geo_col):
    df["lat"] = df[geo_col].map(lambda p: p.x)
    df["long"] = df[geo_col].map(lambda p: p.y)
    return df


def get_hex_centroid(df, hex_code):
    """Get centroid of hexagon
    :param data: dataset
    :type data: dataframe
    :param hex_code: name of column containing the hexagon code
    :type hex_code: string
    :return: coords
    :rtype: list of tuples
    """
    df["hex_centroid"] = df[[hex_code]].apply(
        lambda row: h3.h3_to_geo(row[hex_code]), axis=1
    )
    df["lat"], df["long"] = df["hex_centroid"].str
    return df


def create_geometry(df, lat, long):
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lat], df[long]))
    return df


def get_hex_code(df, lat, long, res):

    df["hex_code"] = df[[lat, long]].apply(
        lambda row: h3.geo_to_h3(row[lat], row[long], res), axis=1
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


def get_hexes_for_ctry(ctry_name="Nigeria", level=7):
    """Get array of all hex codes for specified country

    :param ctry_name: _description_, defaults to 'Nigeria'
    :type ctry_name: str, optional
    :param level: _description_, defaults to 7
    :type level: int, optional
    """
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ctry_shp = world[world.name == ctry_name].geometry.values[0].__geo_interface__
    return h3.polyfill(ctry_shp, level)
