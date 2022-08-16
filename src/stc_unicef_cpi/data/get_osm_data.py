import time
from io import StringIO

import h3.api.numpy_int as h3
import pandas as pd
import requests  # type: ignore
from shapely import wkt

from stc_unicef_cpi.utils.geospatial import (
    get_area_polygon,
    get_hex_code,
    get_hexes_for_ctry,
    get_poly_boundary,
)

# TODO: Consider using OSMnx package instead
# https://osmnx.readthedocs.io/en/stable/
# May be able to acquire other features of interest also


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


def assign_cluster(
    results, country, res, lat="@lat", long="@lon", hex_code_col="hex_code"
):
    """Assign H3 cluster with a specified resolution
    :param results: dataframe with central point of the road lat, lon, length, type_road
    :type results: dataframe
    :param country: country of interest
    :type country: str
    :param res: resolution
    :type res: int
    :returns: hexes with road length
    :rtype: dataframe
    """
    hexes = pd.DataFrame(get_hexes_for_ctry(country, res), columns=[hex_code_col])
    hexes = get_poly_boundary(hexes, hex_code_col)
    results = get_hex_code(results, lat, long, res)
    temp = results.groupby([hex_code_col])["length"].sum().reset_index()
    temp["length_km"] = temp["length"] / 1000
    temp.drop(columns="length", inplace=True)
    hexes = hexes.merge(temp, on=hex_code_col, how="left")
    hexes["length_km"].fillna(0, inplace=True)
    hexes["area_km2"] = hexes["geometry"].apply(lambda poly: get_area_polygon(poly))
    hexes["road_density"] = hexes.apply(
        lambda x: (x["length_km"] / x["area_km2"]), axis=1
    )
    return hexes


def get_road_density(country, res):
    hexes = get_hexes_for_ctry(country, res=2)
    coords = add_neighboring_hexagons(hexes)
    overpass_results = assign_road_length_to_hex(coords)
    road_density = assign_cluster(overpass_results, country, res)

    return road_density
