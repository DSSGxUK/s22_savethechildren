import glob
import os
import pprint
import zipfile
import pandas as pd
import wget
import yaml

from functools import wraps
from time import time

from stc_unicef_cpi.utils.constants import open_cell_colnames


def read_yaml_file(yaml_file):
    """Load yaml configurations"""
    config = None
    try:
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
    except:
        raise FileNotFoundError("Couldn't load the file")

    return config


def get_facebook_credentials(creds_file):
    """Get credentials for accessing FB API from the credentials file"""
    creds = read_yaml_file(creds_file)["facebook"]
    token = creds["access_token"]
    id = creds["account_id"]

    return token, id


def get_open_cell_credentials(creds_file):
    """Get credentials for accessing Open Cell Id from the credentials file"""
    creds = read_yaml_file(creds_file)["open_cell"]
    token = creds["token"]
    return token


class PrettyLog:
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return pprint.pformat(self.obj)


def download_file(url, name):
    """Download a zip file from an specific url

    :param url: URL where the specific object is placed
    :param name: name of output file
    """
    print(url, name)
    wget.download(url, out=name)


def create_folder(dir):
    """Create folder
    :param dir: directory
    :type dir: str
    """
    if not os.path.exists(dir):
        os.mkdir(dir)


def read_csv_gzip(args, colnames=open_cell_colnames):

    df = pd.read_csv(
        args,
        compression="gzip",
        sep=",",
        names=colnames,
        quotechar='"',
        error_bad_lines=False,
        header=None,
    )
    return df


def unzip_file(name):
    """Unzip file
    :param name: name or path of file to unzip
    :type name: str
    """
    name_folder = glob.glob(name)[0].split(".zip")[0]
    with zipfile.ZipFile(glob.glob(name)[0], "r") as h:
        create_folder(name_folder)
        h.extractall(f"{name_folder}/")
    os.remove(glob.glob(name)[0])


def prepend(list, str):
    """Prepend string to elements in list

    :param list: list of elements
    :type list: list
    :param str: string to prepend to each element
    :type str: str
    """
    str += "{0}"
    list = [str.format(element) for element in list]

    return list


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(
            "func:{!r} args:[{!r}, {!r}] took: {:2.4f} sec".format(
                f.__name__, args, kw, te - ts
            )
        )
        return result

    return wrap


@timing
def download_unzip(url, name):

    download_file(url, name)
    unzip_file(name)
