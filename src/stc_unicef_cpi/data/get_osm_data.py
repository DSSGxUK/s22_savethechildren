# -*- coding: utf-8 -*

import pandas as pd
import geopandas as gpd
import h3.api.numpy_int as h3
import requests

from shapely import wkt
from io import StringIO

from src.stc_unicef_cpi.utils.geospatial import get_poly_boundary, get_hexes_for_ctry


def add_neighboring_hexagons(df, hex_code_col="hex_code"):
    """Return dataframe with cluster_id and geometry of hexagons passed in input and their neighbors
    :param data:
    :type data: pandas dataframe
    :return: dataframe with cluster id and geometry of data passed in input plus their neighbors
    :rtype: pandas Dataframe
    """
    # create union of hexagons neighbors with the hexagons in the shapefile

    neighbors = set()
    # create union of hexagons neighbors with the hexagons in the shapefile
    for i in range(df.shape[0]):
        neighbors = neighbors.union(h3.k_ring(df.loc[i][hex_code_col], 1))

    # add hexagons belonging to the dataframe passed as input
    all_hexagons = neighbors.union(set(df[hex_code_col]))

    # add geometry
    # add hexagons belonging to the dataframe passed as input
    data = pd.DataFrame(all_hexagons, columns=[hex_code_col])
    data = gpd.GeoDataFrame(data)
    data = get_poly_boundary(data, hex_code_col)

    return data


def format_polygon(geometry):
    """Format the coordinates of a polygon to pass them to use in a query built with Overpass query.
    The desidered format is: "latitude_1 longitude_1 latitude_2 longitude_2 latitude_3 longitude_3 …"
    :param geometry: string of polygon
    :type geometry: str
    :return: formatted string
    :rtype: str
    """

    # Load the input to extract latitude and longitude
    # The first coordinate is the longitude and the second the latitude
    x, y = wkt.loads(geometry).exterior.coords.xy
    print(len(x))
    polygon_coord_list = [str(y[i]) + " " + str(x[i]) for i in range(6)]
    polygon_coord = " ".join(polygon_coord_list)

    return polygon_coord


def query_osm_road(geometry, elem="way"):
    """Build query to access lat, long, lengths and type of roads in a polygon passed as input.
    The  query will return a csv file
    :param geometry: string of polygon
    :type geometry: str
    :param elem: specify whether you want ways or also nodes and relations (with 'nrw'). Note that nodes have length 0.
    :returns: string of the query
    :rtype: str
    """
    polygon_coord = format_polygon(geometry)
    hw = f"[out:csv(::id, ::lat, ::lon, length, highway)]; {elem} [highway](poly:'"
    geom = f"{hw}{polygon_coord}'); convert result ::=::,::geom=geom(),length=length(); (._;>;); out geom; "

    return geom


def get_osm_info(query_osm_road):
    """Parse query through Overpass API to access Open Street Map data
    :param query_osm_road: string of a query
    :type query_osm_road: str
    :returns: return dataframe with data accessed
    :rtype: dataframe
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={"data": query_osm_road})
    df = pd.read_csv(StringIO(response.text), sep="\t")

    return df


def assign_road_length_to_hex(data_with_neigh):
    """Query the input though Overpass API to access road length through Open Street Map data
    :param data_with_neigh: dataframe with H3 ids in a column called hex_id
    :type data_with_neigh: pandas dataframe
    :returns: return dataframe with lat, lon, length and type of road access through OSM
    :rtype: pandas dataframe
    """
    store_results = get_osm_info(
        query_osm_road(str(data_with_neigh.loc[0]["geometry"]))
    )

    for i in range(1, data_with_neigh.shape[0]):
        temp = get_osm_info(query_osm_road(str(data_with_neigh.loc[i]["geometry"])))
        if temp.shape[1] == 1:
            temp = get_osm_info(query_osm_road(str(data_with_neigh.loc[i]["geometry"])))
        else:
            pass
        store_results = pd.concat([store_results, temp])

    return store_results


def assign_cluster(data, store_results):
    """Assign H3 cluster with a specified resolution to all row of a dataframe (containing in columns '@lat', '@lon' the latitude and longitude)
    :param data: data with hexagonal cluster with resolution
    :type data: pandas dataframe
    :param store_results: dataframe where each row is the central point of the road (lat, lon, length, type of road)
    :type store_results: pandas dataframe
    :returns: return dataframe 'data' with extra column computing road length of that hexagon
    :rtype: pandas dataframe
    """
    # Access resolution of clusters
    resolution = h3.h3_get_resolution(data.loc[0]["hex_id"])
    # Assign cluster to rows
    store_results["cluster_id"] = store_results.progress_apply(
        lambda x: h3.geo_to_h3(x["@lat"], x["@lon"], resolution), axis=1
    )

    # Sum length of road in the same cluster
    temp = store_results.groupby(["cluster_id"])["length"].sum().reset_index()
    # Transform in km
    temp["length_km"] = temp["length"] / 1000
    temp.drop(columns="length", inplace=True)

    data = pd.merge(data, temp, how="left", left_on="hex_id", right_on="cluster_id")
    data["length_km"].fillna(0, inplace=True)

    return data


hex_res_nga = get_hexes_for_ctry(ctry_name="Nigeria", res=2)
hex_code = "hex_code"
df = pd.DataFrame(hex_res_nga, columns=[hex_code])
print(df)
data_with_neigh = add_neighboring_hexagons(df)
print(data_with_neigh)
store_results = assign_road_length_to_hex(data_with_neigh)
print(store_results)
# nga_hex = assign_cluster(nga_hex, store_results)
#
# nga_hex["area_km2"] = nga_hex["geometry"].progress_apply(
#    lambda poly: get_area_polygon(poly)
# )
# nga_hex["road_density"] = nga_hex.progress_apply(
#    lambda x: (x["length_km"] / x["area_km2"]), axis=1
# )
