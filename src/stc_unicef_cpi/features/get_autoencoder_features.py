import pandas as pd
import shutil
import os
import numpy as np

from pathlib import Path
from stc_unicef_cpi.features import autoencoder_features as af


def copy_files(src, trg, word):
    files = os.listdir(src)
    files = [x for x in files if word.lower() in x]
    for file in files:
        shutil.copy2(os.path.join(src, file), trg)


def train_auto_encoder(read_hexes, read_dir, hyper_tunning, save_dir, country, res, thres):
    """Train autoencoder model
    :param read_hexes: _description_
    :type read_hexes: _type_
    :param read_dir: _description_
    :type read_dir: _type_
    :param hyper_tunning: _description_
    :type hyper_tunning: _type_
    :param save_dir: _description_
    :type save_dir: _type_
    :param country: _description_
    :type country: _type_
    :param res: _description_
    :type res: _type_
    :param thres: _description_
    :type thres: _type_
    """
    hexes = pd.read_csv(Path(read_hexes) / f"hexes_{country.lower()}_res{res}_thres{thres}.csv")
    hex_codes = list(hexes.hex_code.values)
    train_img_arr = af.get_train_data(read_dir, hex_codes)
    model_name = f"autoencoder_{country.lower()}_res{res}"
    if hyper_tunning:
        best_hps = af.get_best_hyperparameters(train_img_arr)
        af.get_trained_autoencoder(
            input_data=train_img_arr,
            batch_size=best_hps['batch size'],
            epochs=best_hps['epochs'],
            learning_rate=best_hps['learning rate'],
            save_dir=save_dir,
            model_name=model_name
            )
    else:
        af.get_trained_autoencoder(
            input_data=train_img_arr,
            save_dir=save_dir,
            model_name=model_name
            )


def retrieve_autoencoder_features(read_hexes, trained_autoencoder_dir, country, res, thres, tiff_files_dir):
    """Predict autoencoder features
    :param read_hexes: _description_
    :type read_hexes: _type_
    :param trained_autoencoder_dir: _description_
    :type trained_autoencoder_dir: _type_
    :param country: _description_
    :type country: _type_
    :param res: _description_
    :type res: _type_
    :param thres: _description_
    :type thres: _type_
    :param tiff_files_dir: _description_
    :type tiff_files_dir: _type_
    :return: _description_
    :rtype: _type_
    """
    hexes = pd.read_csv(Path(read_hexes) / f"hexes_{country.lower()}_res{res}_thres{thres}.csv")
    hex_codes = list(hexes.hex_code.values)
    model_name = f"autoencoder_{country.lower()}_res{res}"
    features = af.get_encoded_features(
        trained_autoencoder_dir=trained_autoencoder_dir,
        model_name=model_name,
        hex_codes=np.array(hex_codes),
        tiff_files_dir=tiff_files_dir
        )

    return features


