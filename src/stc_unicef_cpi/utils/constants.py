import inspect
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

# google earth engine parameters
start_ee = "2010-01-01"
end_ee = "2020-01-01"
res_ee = 500
folder_ee = "gee"

current_dir = Path.cwd()
if current_dir.name == "data" and current_dir.parent.name == "stc_unicef_cpi":
    # restrict to when calling from make_dataset.py
    # base directory for data
    base_dir_data = Path.cwd().parent.parent.parent / "data"
    # base directory for autoencoder models
    base_dir_model = Path.cwd().parent.parent.parent / "models"
    # external data
    if Path("/scratch").is_dir():
        base_ext_dir = Path("/scratch/fitzgeraldj")
        base_ext_dir.mkdir(exist_ok=True)
        ext_data = base_ext_dir / "external"
    else:
        ext_data = base_dir_data / "external"
    ext_data.mkdir(exist_ok=True)

    # interim data
    int_data = base_dir_data / "interim"
    int_data.mkdir(exist_ok=True)

    # processed data
    proc_data = base_dir_data / "processed"
    proc_data.mkdir(exist_ok=True)

    # raw data
    raw_data = base_dir_data / "raw"
    raw_data.mkdir(exist_ok=True)

    # tiff files
    if Path("/scratch").is_dir():
        tiff_data = base_ext_dir / "tiff"
    else:
        tiff_data = ext_data / "tiff"
    tiff_data.mkdir(exist_ok=True)

else:
    # just make objects none so no import errors
    # true if not importing from notebook
    base_dir_data = None  # type: ignore
    # base directory for autoencoder models
    base_dir_model = None  # type: ignore
    # external data
    ext_data = None  # type: ignore
    # interim data
    int_data = None  # type: ignore
    # processed data
    proc_data = None  # type: ignore
    # raw data
    raw_data = None  # type: ignore
    # tiff files
    tiff_data = None  # type: ignore
    # importing_file = Path(__file__).name
    # if importing_file == "make_dataset.py":
    # base_dir_data = current_dir / "data"
    # base_dir_model = current_dir / "models"
    # base_dir_data.mkdir(exist_ok=True)
    # base_dir_model.mkdir(exist_ok=True)
    # raise ValueError(
    #     "Must run make_dataset.py from stc_unicef_cpi/data directly for default paths to work as intended: constants.py should only be relevant for this script."
    # )


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
