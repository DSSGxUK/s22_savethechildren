from pathlib import Path

# optimization objective for facebook audience estimates
opt = "REACH"

# names for data retrieved through open cell id
open_cell_colnames = [
    "radio",
    "mcc",
    "mnc",
    "lac",
    "cid",
    "range",
    "long",
    "lat",
    "sample",
    "changeable_1",
    "changeable_0",
    "created",
    "updated",
    "avg_signal",
]

# resolution and area of hexagon in km2
res_area = {
    0: 4250546.8477000,
    1: 607220.9782429,
    2: 86745.8540347,
    3: 12392.2648621,
    4: 1770.3235517,
    5: 252.9033645,
    6: 36.1290521,
    7: 5.1612932,
    8: 0.7373276,
    9: 0.1053325,
    10: 0.0150475,
    11: 0.0021496,
    12: 0.0003071,
    13: 0.0000439,
    14: 0.0000063,
    15: 0.0000009,
}

# google earth engine parameters
start_ee = "2010-01-01"
end_ee = "2020-01-01"
res_ee = 500
folder_ee = "gee"

current_dir = Path.cwd()
if current_dir.name == "data" and current_dir.parent.name == "stc_unicef_cpi":
    # base directory for data
    base_dir_data = Path.cwd().parent.parent.parent / "data"
    base_dir_data.mkdir(exist_ok=True)
    # base directory for autoencoder models
    base_dir_model = Path.cwd().parent.parent.parent / "models"
    base_dir_model.mkdir(exist_ok=True)
else:
    raise ValueError(
        "Must run make_dataset.py from stc_unicef_cpi/data directly for default paths to work as intended"
    )

# external data
ext_data = base_dir_data / "external"
ext_data.mkdir(exist_ok=True)

# interim data
int_data = base_dir_data / "interim"
int_data.mkdir(exist_ok=True)

# raw data
raw_data = base_dir_data / "raw"
raw_data.mkdir(exist_ok=True)

# tiff files

tiff_data = base_dir_data / "tiff"
tiff_data.mkdir(exist_ok=True)

# loggers
str_log = "data_streamer"
dataset_log = "make_dataset.log"

# variables

cols_commuting = [
    "hex_code",
    "name_commuting",
    "win_population_commuting",
    "win_roads_km_commuting",
    "area_commuting",
]

# threshold
cutoff = 30

# speedtest params
serv_type = "mobile"
serv_year = 2021
serv_quart = 4

# models' names
autoencoder_nga = "autoencoder_nga"
