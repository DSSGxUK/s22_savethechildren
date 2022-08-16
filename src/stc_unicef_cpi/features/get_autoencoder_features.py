import pandas as pd
import shutil
import os
import numpy as np

from stc_unicef_cpi.features import autoencoder_features as af


def copy_files(src, trg, word):
    files = os.listdir(src)
    files = [x for x in files if word.lower() in x]
    files = [x for x in files if '.tif' in x]
    for file in files:
        shutil.copy2(os.path.join(src, file), trg)


def train_auto_encoder(hex_codes, read_dir, hyper_tunning, save_dir, country, res):
    """Train autoencoder model
    :param hex_codes: _description_
    :type hex_codes: _type_
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
    """
    train_img_arr = af.get_train_data(read_dir, hex_codes)
    model_name = f"autoencoder_{country.lower()}_res{res}"
    if hyper_tunning:
        best_hps = af.get_best_hyperparameters(train_img_arr)
        af.get_trained_autoencoder(
            input_data=traifn_img_arr,
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


def retrieve_autoencoder_features(hex_codes, trained_autoencoder_dir, country, res, tiff_files_dir, gpu):
    """Predict autoencoder features
    :param hex_codes: _description_
    :type hex_codes: _type_
    :param trained_autoencoder_dir: _description_
    :type trained_autoencoder_dir: _type_
    :param country: _description_
    :type country: _type_
    :param res: _description_
    :type res: _type_
    :param tiff_files_dir: _description_
    :type tiff_files_dir: _type_
    :return: _description_
    :rtype: _type_
    """
    model_name = f"autoencoder_{country.lower()}_res{res}"
    features = af.get_encoded_features(
        trained_autoencoder_dir=trained_autoencoder_dir,
        model_name=model_name,
        hex_codes=np.array(hex_codes),
        tiff_files_dir=tiff_files_dir,
        gpu=gpu
        )

    return features


