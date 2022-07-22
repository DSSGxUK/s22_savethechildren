# -*- coding: utf-8 -*
import pandas as pd
import h3.api.numpy_int as h3
import requests
import time

from shapely import wkt
from io import StringIO

from src.stc_unicef_cpi.utils.geospatial import get_poly_boundary, get_hexes_for_ctry


def format_polygon_coords(geometry):
    """Format the coordinates for Overpass
    :param geometry: string of polygon
    :type geometry: str
    :return: formatted string
    :rtype: str
    """
    x, y = wkt.loads(geometry).exterior.coords.xy
    polygon_coord = [str(y[i]) + " " + str(x[i]) for i in range(len(x) - 1)]
    polygon_coord = " ".join(polygon_coord)

    return polygon_coord


def add_neighboring_hexagons(hex_codes, hex_code_col="hex_code"):
    """Get all hexagons and their respective coordinates
    :param hex_codes: list of hexagons codes
    :type hex_codes: list
    :return: hex_codes and geometry of polygons
    :rtype: list
    """
    neighbors = [h3.k_ring(hex_code) for hex_code in hex_codes]
    neighbors = [x for xs in neighbors for x in xs]
    neighbors.extend(hex_codes)
    hex_codes = list(set(neighbors))
    data = pd.DataFrame(hex_codes, columns=[hex_code_col])
    data = get_poly_boundary(data, hex_code_col)
    coords = data["geometry"].to_list()
    coords = [format_polygon_coords(str(coord)) for coord in coords]
    return coords


def query_osm_road(coords, elem="way"):
    """Build query to access lat, long, lengths and type of roads in a polygon
    :param geometry: string of polygon
    :type geometry: str
    :param elem: specify whether you want ways or also nodes and relations
    :returns: string of the query
    :rtype: str
    """
    hw = f"[out:csv(::id, ::lat, ::lon, length, highway)]; {elem} [highway](poly:'"
    geom = f"{hw}{coords}'); convert result ::=::,::geom=geom(),length=length(); (._;>;); out geom; "

    return geom


def get_osm_info(query_osm_road):
    """Parse query through Overpass to access Open Street
    :param query_osm_road: string of a query
    :type query_osm_road: str
    :returns: return dataframe with data accessed
    :rtype: dataframe
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={"data": query_osm_road})
    df = pd.read_csv(StringIO(response.text), sep="\t")
    time.sleep(5)

    return df


def assign_road_length_to_hex(coords):
    """Query the input though Overpass to get road length
    :param coords: coordinates of polygon
    :type coords: list
    :returns:lat, lon, length and type of highway
    :rtype: dataframe
    """
    data = [get_osm_info(query_osm_road(element)) for element in coords]
    data = pd.concat(data).reset_index(drop=True)
    return data


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


hexes_nigeria = get_hexes_for_ctry(ctry_name="Nigeria", res=2)
coords_nga = add_neighboring_hexagons(hexes_nigeria)
store_results = assign_road_length_to_hex(coords_nga)
# nga_hex = assign_cluster(nga_hex, store_results)
#
# nga_hex["area_km2"] = nga_hex["geometry"].progress_apply(
#    lambda poly: get_area_polygon(poly)
# )
# nga_hex["road_density"] = nga_hex.progress_apply(
#    lambda x: (x["length_km"] / x["area_km2"]), axis=1
# )
