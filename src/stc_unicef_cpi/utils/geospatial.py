# -*- coding: utf-8 -*-
import geopandas as gpd
import h3.api.numpy_int as h3
import math

from pyproj import Geod
from shapely.geometry.polygon import Polygon
from src.stc_unicef_cpi.utils.constants import res_area


def hexagon_radius(res):
    """Get radius according to h3 resolution
    :param res: resolution
    :type res: int
    :return: radius corresponding to the resolution
    :rtype: float
    """
    for key, value in res_area.items():
        if key == res:
            radius = math.sqrt(value * 2 / (3 * math.sqrt(3)))
    return radius


def get_lat_long(data, geo_col):
    """Get latitude and longitude points
    from a given geometry column
    :param data: dataset
    :type data: dataframe
    :param geo_col: name of column containing the geometry
    :type geo_col: string
    :return: dataset
    :rtype: dataframe with latitude and longitude columns
    """
    data["lat"] = data[geo_col].map(lambda p: p.x)
    data["long"] = data[geo_col].map(lambda p: p.y)
    return data


def get_hex_centroid(data, hex_code):
    """Get centroid of hexagon
    :param data: dataset
    :type data: dataframe
    :param hex_code: name of column containing the hexagon code
    :type hex_code: string
    :return: coords
    :rtype: list of tuples
    """
    data["hex_centroid"] = data[[hex_code]].apply(
        lambda row: h3.h3_to_geo(row[hex_code]), axis=1
    )
    data["lat"], data["long"] = data["hex_centroid"].str
    return data


def create_geometry(data, lat, long):
    """Create geometry column from longitude (x) and latitude (y) columns
    :param data: dataset
    :type data: dataframe
    :param lat: name of column containing the longitude of a point
    :type lat: string
    :param long: name of column containing the longitude of a point
    :type long: string
    :return: data
    :rtype: datafrane with geometry column
    """
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[long], data[lat]))
    return data


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


def get_shape_for_ctry(ctry_name):
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ctry_shp = world[world.name == ctry_name]
    return ctry_shp


def get_hexes_for_ctry(ctry_name="Nigeria", res=7):
    """Get array of all hex codes for specified country

    :param ctry_name: _description_, defaults to 'Nigeria'
    :type ctry_name: str, optional
    :param level: _description_, defaults to 7
    :type level: int, optional
    """
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ctry_shp = world[world.name == ctry_name].geometry.values[0].__geo_interface__

    return h3.polyfill(ctry_shp, res)


def get_area_polygon(polygon, crs="WGS84"):
    """Get area of a polygon on earth in km squared
    :param polygon: Polygon
    :type polygon: Polygon
    """
    geometry = wkt.loads(str(polygon))
    # Set CRS to WGS84
    geod = Geod(ellps=crs)
    # Compute the area of the polygon projecting it on earth
    # The area is in meter squared
    area = geod.geometry_area_perimeter(geometry)[0]

    # Transform area in km^2
    area = area / 10**6
    return area


def get_poly_boundary(df, hex_code):

    df["geometry"] = [
        Polygon(h3.h3_to_geo_boundary(x, geo_json=True)) for x in df[hex_code]
    ]
    return df
