"""GET CELL TOWER DATA FROM OPEN CELL ID"""

import glob

import pandas as pd
import requests  # type: ignore
from bs4 import BeautifulSoup

from stc_unicef_cpi.utils.general import (
    create_folder,
    get_open_cell_credentials,
    read_csv_gzip,
)


def get_opencell_url(country, token):
    """get_opencell_url _summary_

    :param country: _description_
    :type country: _type_
    :param token: _description_
    :type token: _type_
    :return: _description_
    :rtype: _type_
    """
    url = f"https://opencellid.org/downloads.php?token={token}"
    country = country.lower()
    soup = BeautifulSoup(requests.get(url).text, "lxml")
    table = soup.find("table", {"id": "regions"})
    t_headers = [th.text.replace("\n", " ").strip() for th in table.find_all("th")]
    countrycol = t_headers[0]
    table_data = []
    for tr in table.tbody.find_all("tr"):
        t_row = {}
        for td, th in zip(tr.find_all("td"), t_headers):
            if t_headers[-1] in th:
                t_row[th] = [a.get("href") for a in td.find_all("a")]
            else:
                t_row[th] = td.text.replace("\n", "").strip()
        table_data.append(t_row)
    df = pd.DataFrame(table_data)
    df[countrycol] = df[countrycol].str.lower()
    if country not in df[countrycol].values:
        print("Invalid country code to get OpenCell Data!")
    else:
        links = df[df[countrycol] == country][t_headers[-1]].values[0]
    return links


def get_cell_data(country, save_path):
    """get_cell_data _summary_

    _extended_summary_

    :param country: _description_
    :type country: _type_
    :param token: _description_
    :type token: _type_
    :return: _description_
    :rtype: _type_
    """
    create_folder(save_path)
    token = get_open_cell_credentials("../../../conf/credentials.yaml")
    urls = get_opencell_url(country, token)
    country = country.lower().replace(" ", "_")
    for url in urls:
        name_file = url.split("file=")[1]
        r = requests.get(url, allow_redirects=True)
        open(f"{save_path}/{country}_{name_file}.tmp", "wb").write(r.content)
    df = pd.concat(map(read_csv_gzip, glob.glob(f"{save_path}/{country}_*.gz.tmp")))
    df = df.reset_index(drop=True)
    return df
