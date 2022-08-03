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
start_ee = '2010-01-01'
end_ee = '2020-01-01'
res_ee = 500
folder_ee = 'gee'

# base directory for data
base_dir_data = '../../../data'

# external data
ext_data = f'{base_dir_data}/external'
