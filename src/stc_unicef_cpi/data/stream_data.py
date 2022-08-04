# -*- coding: utf-8 -*-
"""Data Streaming"""
import os
import logging

import src.stc_unicef_cpi.utils.constants as c
import src.stc_unicef_cpi.data.get_satellite_data as ge


class StreamerObject():

    def __init__(self, country, read_path):
        self.read_path = read_path
        self.country = country


class GoogleEarthEngineStreamer(StreamerObject):

    def __init__(
        self,
        country,
        logging,
        force=False,
        read_path=c.ext_data,
        folder=c.folder_ee,
        res=c.res_ee,
        start=c.start_ee,
        end=c.end_ee
    ):
        super().__init__(country, read_path)
        self.folder = folder
        self.wd = f"{read_path}/{self.folder}/"
        self.res = res
        self.start = start
        self.end = end
        self.force = force
        self.logging = logging
        self.implement()

    def implement(self):

        if self.force:
            self.logging.info(f"Dowloading satellite images of {self.country}...")
            ge.SatelliteImages(self.country, self.folder, self.res, self.start, self.end)
        else:
            file_name = 'cpi_poptotal_' + self.country.lower()+'_500.tif'
            if os.path.exists(self.wd + file_name):
                self.logging.info(f"No need to download Google Earth Engine data! Satellite images of {self.country} are already downloaded.")
            else:
                self.logging.info(f"Downloading satellite images of {self.country}...")
                ge.SatelliteImages(self.country, self.folder, self.res, self.start, self.end)


class RunStreamer(StreamerObject):
    def __init__(self, country, read_path):
        super().__init__(country, read_path)
        self.run()

    def run(self):
        logging.basicConfig(filename='data_streamer.log', format='%(filename)s: %(message)s', level=logging.INFO)
        GoogleEarthEngineStreamer(self.country, logging)