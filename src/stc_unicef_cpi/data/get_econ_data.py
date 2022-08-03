# -*- coding: utf-8 -*-
"""Download econ and facilities data"""
import os
import glob

import src.stc_unicef_cpi.utils.constants as c

from src.stc_unicef_cpi.utils.general import download_unzip, download_file, create_folder, unzip_file, prepend


def get_data_from_calibrated_nighttime(url, out_dir, dir):
    """Get data from calibrated nighttime light data,
    dataset authored by Jiandong Chen, Ming Gao
    :param url: url of data to download
    :type url: str
    :param out_dir: path to output directory
    :type out_dir: str
    :param dir: path to specific data type
    :type dir: str
    """
    out = f"{out_dir}/{dir}.zip"
    download_unzip(url, out)
    zipped = f"{out_dir}/{dir}/*/"
    unzip_file(f"{zipped}2019.zip")
    files = prepend(os.listdir(glob.glob(zipped)[0]), glob.glob(zipped)[0])
    files = [file for file in files if ".zip" in file]
    list(map(os.remove, files))


def download_econ_data(out_dir=c.econ_data):
    """Download economic data
    :param out_dir: path to output directory, defaults to c.econ_data
    :type out_dir: str, optional
    """

    # Check if folder output folder exists
    create_folder(out_dir)

    # Conflict Zones
    conflict_url = "https://ucdp.uu.se/downloads/ged/ged221-csv.zip"
    out_conflict = f"{out_dir}/conflict.zip"
    download_unzip(conflict_url, out_conflict)

    # Critical Infrastructure
    infrastructure_url = "https://zenodo.org/record/4957647/files/CISI.zip?download=1"
    out_infrastructure = f"{out_dir}/infrastructure.zip"
    download_unzip(infrastructure_url, out_infrastructure)

    # Nigeria Health Sites
    health_url = "https://data.humdata.org/dataset/fea18f4e-0463-4194-a21c-602e48e098e1/resource/d09e04f2-1999-4be9-bb50-cb73a1643b37/download/nigeria.csv"
    download_file(health_url, f"{out_dir}/nga_health.csv")

    # Nigeria Education Facilities
    education_url = "https://data.humdata.org/dataset/ec228c18-8edc-4f3c-94c9-a6b946af7229/resource/1a064a21-ffcf-4fb8-a0a6-5cf811d94664/download/nga_education.zip"
    out_education = f"{out_dir}/nga_education.zip"
    download_unzip(education_url, out_education)

    # Global 1 km × 1 km gridded revised real gross domestic product
    gdp_url = "https://figshare.com/ndownloader/files/31456837"
    get_data_from_calibrated_nighttime(gdp_url, out_dir, 'real_gdp')

    # Global 1 km × 1 km gridded revised electricity consumption
    ec_url = "https://figshare.com/ndownloader/files/31456843"
    get_data_from_calibrated_nighttime(ec_url, out_dir, 'elec_cons')

    # Commuting zones
    commuting_url = 'https://data.humdata.org/dataset/b7aaa3d7-cca2-4364-b7ce-afe3134194a2/resource/37c2353d-08a6-4cc2-8364-5fd9f42d6b64/download/data-for-good-at-meta-commuting-zones-july-2021.csv'
    download_file(commuting_url, f"{out_dir}/commuting_zones.csv")


download_econ_data()