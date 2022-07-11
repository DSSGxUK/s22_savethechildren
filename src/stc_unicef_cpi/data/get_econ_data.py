"""Download econ and facilities related data"""

from src.stc_unicef_cpi.utils.general import download_unzip, download_file

# Conflict Zones
conflict_url = "https://ucdp.uu.se/downloads/ged/ged221-csv.zip"
out_conflict = "conflict.zip"
download_unzip(conflict_url, out_conflict)

# Critical Infrastructure
infrastructure_url = "https://zenodo.org/record/4957647/files/CISI.zip?download=1"
out_infrastructure = "critical_infrastructure.zip"
download_unzip(infrastructure_url, out_infrastructure)

# Nigeria Health Sites
health_url = "https://data.humdata.org/dataset/fea18f4e-0463-4194-a21c-602e48e098e1/resource/d09e04f2-1999-4be9-bb50-cb73a1643b37/download/nigeria.csv"
download_file(health_url, "health_sites_nigeria.csv")

# Nigeria Education Facilities
education_url = "https://data.humdata.org/dataset/ec228c18-8edc-4f3c-94c9-a6b946af7229/resource/073ac85c-25f0-49fc-bbfa-9530e5f36e2d/download/nga_education_facilities.xlsx"
download_file(education_url, "education_sites_nigeria.xlsx")
