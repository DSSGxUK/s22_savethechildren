
from datetime import datetime
import geopandas as gpd


def get_speedtest_url(service_type, year, q):

    def quarter_start(year, q):
        if not 1 <= q <= 4:
            raise ValueError("Quarter must be within [1, 2, 3, 4]")

        month = [1, 4, 7, 10]
        return datetime(year, month[q - 1], 1)

    dt = quarter_start(year, q)

    base_url = "https://ookla-open-data.s3.amazonaws.com/shapefiles/performance"
    url = f"{base_url}/type={service_type}/year={dt:%Y}/quarter={q}/{dt:%Y-%m-%d}_performance_{service_type}_tiles.zip"
    name = f"{dt:%Y-%m-%d}_performance_{service_type}_tiles.zip"
    return url, name


def prep_tile(data):
    data = data[['avg_d_kbps', 'avg_u_kbps', 'geometry']]
    print(data)
    print('Writing country speedtest data to directory...')
    data.to_csv('data_speedtest.csv', index=False)


def get_speedtest_info(url):
        try:
            gdf_tiles = gpd.read_file(url)
            data = gdf_tiles
            data = prep_tile(data)
        except:
            pass


url, name = get_speedtest_url(service_type='mobile', year=2021, q=4)
get_speedtest_info(url)
