"""Data Streaming From External Sources"""
import glob as glob
import logging
import os
import warnings

import pandas as pd

import stc_unicef_cpi.data.get_cell_tower_data as cell
import stc_unicef_cpi.data.get_econ_data as econ
import stc_unicef_cpi.data.get_osm_data as osm
import stc_unicef_cpi.data.get_satellite_data as ge
import stc_unicef_cpi.data.get_speedtest_data as speed
import stc_unicef_cpi.utils.constants as c
import stc_unicef_cpi.utils.general as g
import stc_unicef_cpi.utils.geospatial as geo

try:
    import stc_unicef_cpi.data.get_facebook_data as fb
except:
    warnings.warn("Necessary modules for FB data not found - assuming this is not used")


class StreamerObject:
    def __init__(self, country, force, read_path):
        self.read_path = read_path
        self.country = country
        self.force = force


class GoogleEarthEngineStreamer(StreamerObject):
    """Stream data from Google Earth Engine (GEE)"""

    def __init__(
        self,
        country,
        force,
        read_path,
        logging,
        folder=c.folder_ee,
        res=c.res_ee,
        start=c.start_ee,
        end=c.end_ee,
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
            self.logging.info(
                g.PrettyLog(f"Dowloading satellite images of {self.country}...")
            )
            ge.SatelliteImages(
                self.country, self.folder, self.res, self.start, self.end
            )
        else:
            file_name = "cpi_poptotal_" + self.country.lower() + "_500.tif"
            if os.path.exists(self.wd + file_name):
                self.logging.info(
                    g.PrettyLog(
                        f"No need to download Google Earth Engine data! Satellite images of {self.country} are already downloaded."
                    )
                )
            else:
                self.logging.info(
                    g.PrettyLog(f"Downloading satellite images of {self.country}...")
                )
                ge.SatelliteImages(
                    self.country, self.folder, self.res, self.start, self.end
                )


class EconomicStreamer(StreamerObject):
    """Stream economic variables if not downloaded"""

    def __init__(self, country, logging, read_path, force):
        super().__init__(country, force, read_path)
        self.logging = logging
        self.implement()

    def implement(self):

        if self.force:
            self.logging.info(
                g.PrettyLog(f"Downloading economic data for {self.country}...")
            )
            econ.download_econ_data(self.read_path)
        else:
            file_name = "gdp_ppp_30.nc"
            if os.path.exists(self.read_path + file_name):
                self.logging.info(
                    g.PrettyLog(
                        f"No need to download economic data! Economic data for {self.country} is already downloaded."
                    )
                )
            else:
                self.logging.info(f"Downloading economic data for {self.country}...")
                econ.download_econ_data(self.read_path)


class FacebookMarketingStreamer(StreamerObject):
    """Stream data from Facebook Marketing Api"""

    def __init__(self, country, force, read_path, res, logging):
        super().__init__(country, force, read_path)
        self.logging = logging
        self.res = res
        self.implement()

    def implement(self):
        hexes = geo.get_hexes_for_ctry(self.country, self.res)
        ctry = pd.DataFrame(hexes, columns=["hex_code"])
        coords = geo.get_hex_centroid(ctry, "hex_code")["hex_centroid"].values
        file_name = "fb_aud_" + self.country.lower() + "_res" + self.res + ".parquet"
        if self.force:
            self.logging.info(
                g.PrettyLog(f"Retrieving audience estimates for {self.country}...")
            )
            fb.get_facebook_estimates(coords, self.read_path, file_name, self.res)

        else:
            if os.path.exists(f"{self.read_path}/{file_name}"):
                self.logging.info(
                    g.PrettyLog(
                        f"No need to retrieve audience estimates! Estimates for {self.country} are already downloaded."
                    )
                )
            else:
                self.logging.info(
                    g.PrettyLog(f"Dowloading audience estimates for {self.country}...")
                )
                fb.get_facebook_estimates(coords, self.read_path, file_name, self.res)


class RoadDensityStreamer(StreamerObject):
    """Stream data from Open Street Map"""

    def __init__(self, country, force, read_path, res, logging):
        super().__init__(country, force, read_path)
        self.logging = logging
        self.res = res
        self.implement()

    def implement(self):
        file_name = (
            "road_density_" + self.country.lower() + "_res" + str(self.res) + ".csv"
        )
        if self.force:
            self.logging.info(
                g.PrettyLog(f"Retrieving road density estimates for {self.country}...")
            )
            rd = osm.get_road_density(self.country, self.res)
            rd.to_csv(f"{self.read_path}/{file_name}", index=False)
        else:
            if os.path.exists(f"{self.read_path}/{file_name}"):
                self.logging.info(
                    g.PrettyLog(
                        f"No need to retrieve road density estimates! Estimates for {self.country} are already downloaded."
                    )
                )
            else:
                self.logging.info(
                    g.PrettyLog(
                        f"Retrieving road density estimates for {self.country}..."
                    )
                )
                rd = osm.get_road_density(self.country, self.res)
                print(rd)
                rd.to_csv(f"{self.read_path}/{file_name}", index=False)


class SpeedTestStreamer(StreamerObject):
    """Stream data from Open Street Map"""

    def __init__(
        self,
        country,
        force,
        read_path,
        logging,
        service_type=c.serv_type,
        year=c.serv_year,
        q=c.serv_quart,
    ):
        super().__init__(country, force, read_path)
        self.logging = logging
        self.service_type = service_type
        self.year = year
        self.q = q
        self.implement()

    def implement(self):
        file_name = "2021-10-01_performance_mobile_tiles.csv"
        if self.force:
            self.logging.info(
                g.PrettyLog(f"Speed test data estimates for {self.country}...")
            )
            url, name = speed.get_speedtest_url(self.service_type, self.year, self.q)
            speed.get_speedtest_info(url, name, self.read_path)
        else:
            if os.path.exists(f"{self.read_path}/{file_name}"):
                self.logging.info(
                    g.PrettyLog(
                        f"No need to retrieve speed test data estimates! Estimates for {self.country} are already downloaded."
                    )
                )
            else:
                self.logging.info(
                    g.PrettyLog(
                        f"Retrieving speed test data estimates for {self.country}..."
                    )
                )
                url, name = speed.get_speedtest_url(
                    self.service_type, self.year, self.q
                )
                speed.get_speedtest_info(url, name, self.read_path)


class OpenCellStreamer(StreamerObject):
    """Stream data from Open Street Map"""

    def __init__(self, country, force, read_path, logging):
        super().__init__(country, force, read_path)
        self.logging = logging
        self.implement()

    def implement(self):
        file_name = f"{self.country.lower()}_*.csv.gz.tmp"
        if self.force:
            self.logging.info(
                g.PrettyLog(f"Retrieving open cell id data for {self.country}...")
            )
            cell.get_cell_data(self.country, self.read_path)
        else:
            if glob.glob(f"{self.read_path}/{file_name}"):
                self.logging.info(
                    g.PrettyLog(
                        f"No need to retrieve pen cell id data! Estimates for {self.country} are already downloaded."
                    )
                )
            else:
                self.logging.info(
                    g.PrettyLog(f"Retrieving open cell id data for {self.country}...")
                )
                cell.get_cell_data(self.country, self.read_path)


class RunStreamer(StreamerObject):
    def __init__(
        self, country, res, force, audience, read_path=c.ext_data, name_logger=c.str_log
    ):
        super().__init__(country, force, read_path)
        self.name_logger = name_logger
        self.res = res
        self.audience = audience
        self.stream()

    def stream(self):
        logging.basicConfig(
            filename=f"{self.name_logger}.log",
            format="%(filename)s: %(message)s",
            level=logging.INFO,
        )
        # GoogleEarthEngineStreamer(self.country, self.force, self.read_path, logging)
        # RoadDensityStreamer(self.country, self.force, self.read_path, self.res, logging)
        # SpeedTestStreamer(self.country, self.force, self.read_path, logging)
        EconomicStreamer(self.country, logging, self.read_path, self.force)
        # OpenCellStreamer(self.country, self.force, self.read_path, logging)
        if self.audience:
            FacebookMarketingStreamer(
                self.country, self.force, self.read_path, self.res, logging
            )
