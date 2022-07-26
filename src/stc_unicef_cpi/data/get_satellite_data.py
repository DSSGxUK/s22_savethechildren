# -*- coding: utf-8 -*-

import ee


def init_gee():
    """Authenticate the Google earth engine with google account

    _extended_summary_
    """
    ee.Initialize()


init_gee()


def downloader_google_earth_engine(ee_object, region, scale):
    """Download ImageCollection or single image
    :param ee_object: _description_
    :type ee_object: _type_
    :param region: _description_
    :type region: _type_
    :param scale: _description_
    :type scale: _type_
    :return: _description_
    :rtype: _type_
    """
    init_gee()
    try:
        if isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_object = ee_object.mosaic()
        url = ee_object.getDownloadUrl(
            {"scale": scale, "crs": "EPSG:4326", "region": region}
        )
        return url
    except:
        print("Could not download")
