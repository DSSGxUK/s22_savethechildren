from pathlib import Path

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask


def netcdf_to_clipped_array(file_path, *, ctry_name, save_dir=None, plot=False):
    """Read netCDF file and return either array clipped to
    specified country, or a GeoTIFF clipped to this country
    and saved in the specified directory with same name as
    before

    :param file_path: _description_
    :type file_path: _type_
    :param save_dir: _description_, defaults to None
    :type save_dir: _type_, optional
    :param ctry_name: _description_
    :type ctry_name: str, optional
    :param plot: _description_, defaults to True
    :type plot: bool, optional
    :return: _description_
    :rtype: _type_
    """
    fname = Path(file_path).name
    # world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    # use high res version to avoid clipping
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    world = reader.records()
    with rasterio.open(f"netcdf:{file_path}", "r", masked=True) as netf:
        ctry_shp = next(
            filter(lambda x: x.attributes["NAME"] == ctry_name, world)
        ).geometry
        if netf.crs is not None and netf.crs != "EPSG:4326":
            # NB assumes that no CRS corresponds to EPSG:4326 (as standard)
            ctry_shp = gpd.GeoSeries(ctry_shp)
            ctry_shp.crs = "EPSG:4326"
            ctry_shp = ctry_shp.to_crs(netf.crs).geometry
        print(
            f"Pixel scale in crs {netf.crs}: {netf.res}"
        )  # shows pixel scale in crs units
        # print(netf.window(*nga_bbox))
        # nga_subset = netf.read(window=netf.window(*nga_bbox))
        # print(nga_subset[-1].min(),nga_subset[-1].max())
        # plt.imshow(np.log(nga_subset[-1,:,:]+10),cmap='PiYG')
        # plt.show()
        try:
            out_image, out_transform = rasterio.mask.mask(netf, ctry_shp, crop=True)
        except TypeError:
            # polygon not iterable
            out_image, out_transform = rasterio.mask.mask(netf, [ctry_shp], crop=True)
        if plot:
            plt.imshow(
                np.log(out_image[-1, :, :] - out_image[-1].min() + 1), cmap="PiYG"
            )
            plt.show()
        out_meta = netf.meta

    if save_dir is not None:
        save_dir = Path(save_dir)
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rasterio.open(
            save_dir / (ctry_name.lower() + "_" + fname.rstrip(".nc") + ".tif"),
            "w",
            **out_meta,
        ) as dest:
            dest.write(out_image)
    else:
        return out_image
