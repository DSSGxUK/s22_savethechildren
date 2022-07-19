# -*- coding: utf-8 -*-
import wget
import zipfile
import os
import yaml

from functools import wraps
from time import time


def read_yaml_file(yaml_file):
    """Load yaml configurations"""
    config = None
    try:
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
    except:
        raise FileNotFoundError("Couldnt load the file")

    return config


def get_facebook_credentials(creds_file):
    """Get credentials for accessing FB API from the credentials file"""
    creds = read_yaml_file(creds_file)["facebook"]
    token = creds["access_token"]
    id = creds["account_id"]
    optimization = creds["optimization"]

    return token, id, optimization


def download_file(url, name):
    """Download a zip file from an specific url

    :param url: URL where the specific object is placed
    :param name: name of output file
    """
    print(url, name)
    wget.download(url, out=name)


def create_folder(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def unzip_file(name):
    name_folder = name.split(".zip")[0]
    with zipfile.ZipFile(name, "r") as h:
        create_folder(name_folder)
        h.extractall(f"{name_folder}/")
    os.remove(name)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


@timing
def download_unzip(url, name):

    download_file(url, name)
    unzip_file(name)
