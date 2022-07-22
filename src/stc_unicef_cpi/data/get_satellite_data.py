# -*- coding: utf-8 -*-

import ee


def init_gee():
    """Authenticate the Google earth engine with google account

    _extended_summary_
    """
    ee.Initialize()


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
        # download image
        if isinstance(ee_object, ee.image.Image):
            print("Its Image")
            url = ee_object.getDownloadUrl(
                {"scale": scale, "crs": "EPSG:4326", "region": region}
            )
            return url

        # download imagecollection
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            print("It's ImageCollection")
            ee_object_new = ee_object.mosaic()
            url = ee_object_new.getDownloadUrl(
                {"scale": scale, "crs": "EPSG:4326", "region": region}
            )
            return url
    except:
        print("Could not download")
