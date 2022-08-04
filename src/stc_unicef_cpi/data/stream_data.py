# -*- coding: utf-8 -*-
"""Data Streaming From External Sources"""
import os
import logging

import src.stc_unicef_cpi.utils.constants as c
import src.stc_unicef_cpi.data.get_satellite_data as ge
import src.stc_unicef_cpi.data.get_econ_data as econ
import src.stc_unicef_cpi.utils.general as g


class StreamerObject():

    def __init__(self, country, force=False, read_path=c.ext_data):
        self.read_path = read_path
        self.country = country
        self.force = force


class GoogleEarthEngineStreamer(StreamerObject):
    """Stream data from Google Earth Engine (GEE)"""
    def __init__(
        self, country, force, read_path, logging, folder=c.folder_ee, res=c.res_ee, start=c.start_ee, end=c.end_ee
    ):
        super().__init__(country, force, read_path)
        self.logging = logging
        self.folder = folder
        self.wd = f"{read_path}/{folder}/"
        self.res = res
        self.start = start
        self.end = end
        self.implement()

    def implement(self):
        if self.force:
            self.logging.info(g.PrettyLog(f"Dowloading satellite images of {self.country}..."))
            ge.SatelliteImages(self.country, self.folder, self.res, self.start, self.end)
        else:
            file_name = 'cpi_poptotal_' + self.country.lower()+'_500.tif'
            if os.path.exists(self.wd + file_name):
                self.logging.info(g.PrettyLog(f"No need to download Google Earth Engine data! Satellite images of {self.country} are already downloaded."))
            else:
                self.logging.info(g.PrettyLog(f"Downloading satellite images of {self.country}..."))
                ge.SatelliteImages(self.country, self.folder, self.res, self.start, self.end)


class EconomicStreamer(StreamerObject):

    def __init__(
        self,
        country,
        logging,
        read_path,
        force
    ):
        super().__init__(country, read_path, force)
        self.logging = logging
        self.implement()

    def implement(self):

        if self.force:
            self.logging.info(g.PrettyLog(f"Downloading economic data for {self.country}..."))
            econ.download_econ_data(self.read_path)
        else:
            file_name = 'gdp_ppp_30.nc'
            if os.path.exists(self.read_path + file_name):
                self.logging.info(g.PrettyLog(f"No need to download economic data! Economic data for {self.country} is already downloaded."))
            else:
                self.logging.info(f"Downloading economic data for {self.country}...")
                econ.download_econ_data(self.read_path)


class RunStreamer(StreamerObject):
    def __init__(self, country, force, read_path, name_logger=c.str_log):
        super().__init__(country, force, read_path)
        self.name_logger = name_logger
        self.run()

    def run(self):
        logging.basicConfig(filename=f'{self.name_logger}.log', format='%(filename)s: %(message)s', level=logging.INFO)
        GoogleEarthEngineStreamer(self.country, self.force, self.read_path, logging)
        #EconomicStreamer(self.country, logging)


RunStreamer(country='Senegal')
