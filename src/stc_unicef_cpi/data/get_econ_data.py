# -*- coding: utf-8 -*-
"""Download econ and facilities related data"""
import os
import src.stc_unicef_cpi.utils.constants as c

from src.stc_unicef_cpi.utils.general import download_unzip, download_file, create_folder, unzip_file, prepend


def download_econ_data(out_dir=c.econ_data):
    """Download Economical Data"""

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
    out_gdp = f"{out_dir}/real_gdp.zip"
    download_unzip(gdp_url, out_gdp)
    gdp_zip = f"{out_dir}/real_gdp/updated real GDP/"
    gdp_files = prepend(os.listdir(gdp_zip), gdp_zip)
    # unzip all elements in folder
    list(map(unzip_file, gdp_files))


download_econ_data()