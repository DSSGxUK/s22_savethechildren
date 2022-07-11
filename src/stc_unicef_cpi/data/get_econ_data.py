from src.stc_unicef_cpi.utils.general import download_unzip

conflict_url = "https://ucdp.uu.se/downloads/ged/ged221-csv.zip"
out_conflict = "conflict.zip"
download_unzip(conflict_url, out_conflict)
