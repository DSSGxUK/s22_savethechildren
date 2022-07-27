from pathlib import Path
from typing import List, Union

import keras
import numpy as np
import rasterio

import stc_unicef_cpi.data.process_geotiff as pg

# TODO: finish CV splitting fn


class KerasDataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        hex_idxs: np.ndarray,
        labels: np.ndarray,
        batch_size=32,
        dim=(16, 16),
        data_files: Union[List[str], List[Path]] = None,
        # n_classes=10,
        shuffle=True,
    ):
        "Initialization"
        try:
            assert len(dim) == 2
            self.dim = dim
        except AssertionError:
            raise ValueError(
                "dim must be a tuple of length 2, specifying height and width of extracted images"
            )
        self.batch_size = batch_size
        self.labels = labels
        self.hex_idxs = hex_idxs
        # self.idxs = np.arange(len(self.hex_idxs))
        data_files = list(map(Path, data_files))  # type: ignore
        # assume GeoTIFF files listed, and end in '.tif'
        self.tif_files = [file for file in data_files if file.suffix == ".tif"]
        # assume npy files are only other type of data, and the order
        # matches the order of hex_idxs - this allows memory mapped
        # reading from disk
        self.np_files = [file for file in data_files if file.suffix == ".npy"]
        try:
            assert set(self.tif_files).union(set(self.np_files)) == set(data_files)
        except AssertionError:
            raise ValueError(
                "data_files must be a list of GeoTIFF (.tif) and npy files only"
            )
        if len(self.np_files) > 0:
            raise NotImplementedError(
                "npy files not yet implemented - currently just loader for tif files for DL pipeline"
            )
        # np.load(file,mmap_mode='r')

        nbands = 0
        for tif_file in self.tif_files:
            with rasterio.open(tif_file) as open_file:
                nbands += len(open_file.indexes)
        self.n_channels = nbands
        # self.n_classes = n_classes
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        NB floor used so that at final step, (idx+1)*batch_size < len(hex_idxs)
        but this means that if shuffle=False, the final (incomplete) batch
        of data will never be seen.
        """
        return int(np.floor(len(self.hex_idxs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        idxs = self.idxs[index * self.batch_size : (index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(idxs)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.idxs = np.arange(len(self.hex_idxs))
        if self.shuffle == True:
            np.random.shuffle(self.idxs)

    def __data_generation(self, idxs):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        hex_idxs = self.hex_idxs[idxs]
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=self.labels.dtype)

        # Generate data
        # by default in shape (image_idx,band,i,j), so permute to (idx,i,j,band)
        X = pg.extract_ims_from_hex_codes(
            self.tif_files, hex_idxs, width=self.dim[1], height=self.dim[0]
        ).transpose((0, 2, 3, 1))
        y = self.labels[idxs]

        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


def cv_split(all_hex_idxs: np.ndarray, labels: np.ndarray, k: int, mode="normal"):
    """Generate k folds on (fixed order) hex dataset - either
    fully random (normal), stratified by interval (stratified),
    or spatially (spatial)

    :param all_hex_idxs: _description_
    :type all_hex_idxs: np.ndarray of type int
    :param labels: corresponding target labels for these idxs
    :type labels: np.ndarray of type int
    :param k: _description_
    :type k: int
    :param mode: _description_, defaults to 'normal'
    :type mode: str, optional
    :return: _description_
    :rtype: _type_
    """
    pass


# """Example Keras script:

# import numpy as np

# from keras.models import Sequential
# from my_classes import DataGenerator

# # Parameters
# params = {'dim': (32,32,32),
#           'batch_size': 64,
#           'n_classes': 6,
#           'n_channels': 1,
#           'shuffle': True}

# # Datasets
# partition = # IDs
# labels = # Labels

# # Generators
# training_generator = KerasDataGenerator(partition['train'], labels, **params)
# validation_generator = KerasDataGenerator(partition['validation'], labels, **params)

# # Design model
# model = Sequential()
# [...] # Architecture
# model.compile()

# # Train model on dataset
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=6)


# NB for Keras need to setup K-fold CV (of whatever kind) manually, e.g.

# def load_data_kfold(k):

#     train = pd.read_json('../input/train.json')
#     train.inc_angle = train.inc_angle.replace('na', 0)

#     x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
#     x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
#     x_band3 = x_band1 / x_band2

#     X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
#                             , x_band2[:, :, :, np.newaxis]
#                             , x_band3[:, :, :, np.newaxis]], axis=-1)

#     y_train = np.array(train["is_iceberg"])

#     folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X_train, y_train))

#     return folds, X_train, y_train

# k = 7
# folds, X_train, y_train = load_data_kfold(k)

# ...

# for j, (train_idx, val_idx) in enumerate(folds):

#     print('\nFold ',j)
#     X_train_cv = X_train[train_idx]
#     y_train_cv = y_train[train_idx]
#     X_valid_cv = X_train[val_idx]
#     y_valid_cv= y_train[val_idx]

#     name_weights = "final_model_fold" + str(j) + "_weights.h5"
#     callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
#     generator = gen.flow(X_train_cv, y_train_cv, batch_size = batch_size)
#     model = get_model()
#     model.fit_generator(
#                 generator,
#                 steps_per_epoch=len(X_train_cv)/batch_size,
#                 epochs=15,
#                 shuffle=True,
#                 verbose=1,
#                 validation_data = (X_valid_cv, y_valid_cv),
#                 callbacks = callbacks)

#     print(model.evaluate(X_valid_cv, y_valid_cv))

# """
