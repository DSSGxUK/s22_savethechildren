import geopandas as gpd
import stc_unicef_cpi.utils.general as g

from datetime import datetime
from pathlib import Path


def get_speedtest_url(service_type, year, q):
    """Get Speed Test Url From Ookla
    :param service_type: type of network performance
    :type service_type: str
    :param year: year
    :type year: int
    :param q: quarter
    :type q: int
    """

    def quarter_start(year, q):
        if not 1 <= q <= 4:
            raise ValueError("Quarter must be within [1, 2, 3, 4]")

        month = [1, 4, 7, 10]
        return datetime(year, month[q - 1], 1)

    dt = quarter_start(year, q)

    base_url = "https://ookla-open-data.s3.amazonaws.com/shapefiles/performance"
    url = f"{base_url}/type={service_type}/year={dt:%Y}/quarter={q}/{dt:%Y-%m-%d}_performance_{service_type}_tiles.zip"
    name = f"{dt:%Y-%m-%d}_performance_{service_type}_tiles.csv"
    return url, name


def prep_tile(data, name, path_save):
    """Prepare tile

    _extended_summary_

    :param data: _description_
    :type data: _type_
    :param name: _description_
    :type name: _type_
    """
    data = data[['avg_d_kbps', 'avg_u_kbps', 'geometry']]
    print('Saving speedtest data to directory...')
    g.create_folder(Path(path_save) / "connectivity")
    data.to_csv(Path(path_save) / "connectivity" / f"{name}", index=False)


def get_speedtest_info(url, name, path_save):
        try:
            gdf_tiles = gpd.read_file(url)
            data = gdf_tiles
            data = prep_tile(data, name, path_save)
        except:
            raise ValueError("Unable to retrieve data.")
