# -*- coding: utf-8 -*

import pandas as pd
import geopandas as gpd
import h3
import requests
import tqdm

from shapely.geometry.polygon import Polygon
from shapely.geometry import mapping
from shapely import wkt
from shapely.ops import unary_union
from io import StringIO
from tqdm.auto import tqdm

from src.stc_unicef_cpi.utils.geospatial import get_area_polygon


def hexagonal_grid(shp_gdf, resolution, crs=4326):
    """Return an hexagonal grid of a region with a specified resolution passed in input
    :param shp_gdf:
    :type shp_gdf: shp
    :param resolution: resolution of H3 hexagonal grid (integer between 1 and 15)
    :type resolution: int
    :return: dataframe with cluster id and geometry
    :rtype: pandas Dataframe
    """

    gdf = shp_gdf.to_crs(epsg=crs)

    # Get union of the shape
    union_poly = unary_union(gdf.geometry)

    # Find the hexagons within the shape boundary using PolyFill
    hex_list = []
    for n, g in enumerate(union_poly):
        if (n + 1) % 100 == 0:
            print(str(n + 1) + "/" + str(len(union_poly)))
        temp = mapping(g)
        temp["coordinates"] = [[[j[1], j[0]] for j in i] for i in temp["coordinates"]]
        hex_list.extend(h3.polyfill(temp, res=resolution))

    # Create hexagon data frame
    nga_hex_res = pd.DataFrame(hex_list, columns=["hex_id"])
    print("Shape: " + str(nga_hex_res.shape))

    # Create hexagon geometry and GeoDataFrame
    # If True, return output in GeoJson format: lng/lat pairs (opposite order), and have the last pair be the same as the first.
    nga_hex_res["geometry"] = [
        Polygon(h3.h3_to_geo_boundary(x, geo_json=True)) for x in nga_hex_res["hex_id"]
    ]
    nga_hex_res = gpd.GeoDataFrame(nga_hex_res)

    return nga_hex_res


def add_neighboring_hexagons(data):
    """Return dataframe with cluster_id and geometry of hexagons passed in input and their neighbors
    :param data:
    :type data: pandas dataframe
    :return: dataframe with cluster id and geometry of data passed in input plus their neighbors
    :rtype: pandas Dataframe
    """

    neighbors = set()
    # create union of hexagons neighbors with the hexagons in the shapefile
    for i in range(data.shape[0]):
        neighbors = neighbors.union(h3.k_ring(data.loc[i]["hex_id"], 1))

    # add hexagons belonging to the dataframe passed as input
    all_hexagons = neighbors.union(set(data["hex_id"]))

    # add geometry
    data_with_neigh = pd.DataFrame(all_hexagons, columns=["hex_id"])
    data_with_neigh["geometry"] = [
        Polygon(h3.h3_to_geo_boundary(x, geo_json=True))
        for x in data_with_neigh["hex_id"]
    ]
    data_with_neigh = gpd.GeoDataFrame(data_with_neigh)

    return data_with_neigh


"""# Road Length"""


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

    # Create the string (attention: first put y that is the latitude)
    polygon_coord_list = [str(y[i]) + " " + str(x[i]) for i in range(6)]
    polygon_coord = " ".join(polygon_coord_list)

    return polygon_coord


def build_query(geometry, elem="way"):
    """Build query to access lat, long, lengths and type of roads in a polygon passed as input.
    The  query will return a csv file
    :param geometry: string of polygon
    :type geometry: str
    :param elem: specify whether you want ways or also nodes and relations (with 'nrw'). Note that nodes have length 0.
    :returns: string of the query
    :rtype: str
    """

    # Format correctly the polygon to pass to the query
    polygon_coord = format_polygon(geometry)

    # Build query
    query_1 = (
        "[out:csv(::id, ::lat, ::lon, length, highway)];   "
        + str(elem)
        + "[highway](poly:' "
    )
    query_2 = (
        "'); convert result ::=::,::geom=geom(),length=length(); (._;>;);   out geom; "
    )

    return query_1 + str(polygon_coord) + query_2


def get_road(build_query):
    """Query the input though Overpass API to access Open Street Map data
    :param build_query: string of a query
    :type build_query: str
    :returns: return dataframe with data accessed
    :rtype: pandas dataframe
    """
    overpass_url = "http://overpass-api.de/api/interpreter"  # url of overpass api
    response = requests.get(
        overpass_url, params={"data": build_query}
    )  # sending a get request and passing the overpass query as data parameter in url

    # the length is in meters
    df = pd.read_csv(StringIO(response.text), sep="\t")
    print(df.shape)

    return df


def assign_road_length_to_hex(data_with_neigh):
    """Query the input though Overpass API to access road length through Open Street Map data
    :param data_with_neigh: dataframe with H3 ids in a column called hex_id
    :type data_with_neigh: pandas dataframe
    :returns: return dataframe with lat, lon, length and type of road access through OSM
    :rtype: pandas dataframe
    """

    # Get the length of the road in meters
    store_results = get_road(build_query(str(data_with_neigh.loc[0]["geometry"])))

    # Sometimes the API call failed, so I save where it fails in wrongs and I call it again
    wrongs = []
    for i in tqdm(range(1, data_with_neigh.shape[0])):
        temp = get_road(build_query(str(data_with_neigh.loc[i]["geometry"])))
        if temp.shape[1] == 1:
            wrongs.append(i)
        else:
            store_results = pd.concat([store_results, temp])

    print("Wrongs: " + str(wrongs))
    for i in tqdm(wrongs):
        temp = get_road(build_query(str(data_with_neigh.loc[i]["geometry"])))
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


"""# Road Density"""

country = "Nigeria"
shp_ctry = get_shape_for_ctry(country)


# Pass resolution = 2 so that there are not too many hexagons
nga_hex_res_2 = hexagonal_grid(gdf, resolution=2)
data_with_neigh = add_neighboring_hexagons(nga_hex_res_2)

store_results = assign_road_length_to_hex(data_with_neigh)
nga_hex = assign_cluster(nga_hex, store_results)

nga_hex["Area_km2"] = nga_hex["geometry"].progress_apply(lambda x: get_area_polygon(x))
nga_hex["Road_density"] = nga_hex.progress_apply(
    lambda x: (x["length_km"] / x["Area_km2"]), axis=1
)
