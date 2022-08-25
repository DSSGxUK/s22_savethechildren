import warnings
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterator, List, Tuple, Union

import h3.api.numpy_int as h3
import numpy as np
import pandas as pd
import rasterio
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import column_or_1d
from tensorflow import keras

import stc_unicef_cpi.data.process_geotiff as pg


class KerasDataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        hex_idxs: np.ndarray,
        # labels: np.ndarray,
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
        # self.labels = labels
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
        return int(np.ceil(len(self.hex_idxs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        idxs = self.idxs[index * self.batch_size : (index + 1) * self.batch_size]
        # Generate data
        X = self.__data_generation(idxs)

        return X

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
        # y = np.empty((self.batch_size), dtype=self.labels.dtype)

        # Generate data
        # by default in shape (image_idx,band,i,j), so permute to (idx,i,j,band)
        X = pg.extract_ims_from_hex_codes(
            self.tif_files, hex_idxs, width=self.dim[1], height=self.dim[0]
        ).transpose((0, 2, 3, 1))
        # y = self.labels[idxs]

        shape = X.shape
        X = StandardScaler().fit_transform(X.reshape(X.shape[0], -1))
        X = X.reshape(shape)

        if np.isnan(np.sum(X)):
            min = np.min(X[~np.isnan(X)])
            if min < 0:
                X = np.nan_to_num(X, nan=2 * min)
            else:
                X = np.nan_to_num(X, nan=-2 * min)

        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X  # , y


def cv_split(
    all_hex_idxs: np.ndarray,
    labels: np.ndarray,
    k: int,
    mode="normal",
    seed=42,
    strat_cuts=5,
):
    """Generate k folds on (fixed order) hex dataset - either
    fully random (normal), stratified by interval (stratified),
    or spatially (spatial)
    :param all_hex_idxs: Array of hex codes of dataset
    :type all_hex_idxs: np.ndarray of type int
    :param labels: corresponding target labels for these idxs
    :type labels: np.ndarray of type int
    :param k: Number of folds
    :type k: int
    :param mode: mode to generate folds, choice of ['normal','stratified','spatial']. Defaults to 'normal' (fully random)
    :type mode: str, optional
    :param seed: random seed, defaults to 42
    :type seed: int, optional
    :param strat_cuts: number of intervals to cut the data into for stratified CV, defaults to 5
    :type strat_cuts: int, optional
    :return: folds
    :rtype: _type_
    """
    try:
        assert len(all_hex_idxs) == len(labels)
    except AssertionError:
        raise ValueError("all_hex_idxs and labels must be the same length")
    if mode == "normal":
        return KFold(n_splits=k, random_state=seed).split(all_hex_idxs, labels)
    elif mode == "stratified":
        strat_labels = pd.cut(labels, strat_cuts, labels=False)
        return StratifiedKFold(n_splits=k, random_state=seed).split(
            all_hex_idxs, strat_labels
        )
    elif mode == "spatial":
        return HexSpatialKFold(n_splits=k, random_state=seed).split(
            all_hex_idxs, labels
        )


class HexSpatialKFold(KFold):
    """NB lightly modified version of GroupKFold
    - new code takes hex codes passed and generates n_split suitable groups, rather than
    requiring these to be passed along with X, y, as in original GroupKFold
    """

    def __init__(self, n_splits=5, *, random_state=None, hex_idx=None):
        super().__init__(n_splits=n_splits)
        self.random_state = random_state
        self.hex_idx = hex_idx

    def _iter_test_indices(self, X, y, groups=None):
        if groups is None:
            groups = self.get_spatial_groups(X)
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups)

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Distribute samples by adding the largest weight to the lightest fold
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def haversine(self, latlon1, latlon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1 = latlon1[::-1]
        lon2, lat2 = latlon2[::-1]
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in km. Use 3956 for miles
        return c * r

    def get_even_clusters(self, X, n_clusters):
        cluster_size = int(np.ceil(len(X) / n_clusters))
        kmeans = KMeans(n_clusters, random_state=self.random_state)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        centers = (
            centers.reshape(-1, 1, X.shape[-1])
            .repeat(cluster_size, 1)
            .reshape(-1, X.shape[-1])
        )
        distance_matrix = cdist(X, centers, metric=self.haversine)
        clusters = linear_sum_assignment(distance_matrix)[1] // cluster_size
        return clusters

    def get_spatial_groups(self, X):
        if len(X.shape) == 1:
            latlongs = np.array([h3.h3_to_geo(hex_code) for hex_code in X])
        else:
            # assume passing pandas df with columns named 'hex_code'
            try:
                if self.hex_idx is None:
                    latlongs = np.array(
                        [h3.h3_to_geo(hex_code) for hex_code in X["hex_code"]]
                    )
                else:
                    try:
                        latlongs = np.array(
                            [h3.h3_to_geo(hex_code) for hex_code in X[:, self.hex_idx]]
                        )
                    except TypeError:
                        print(X[:5, self.hex_idx])
                        raise ValueError(
                            "Column indiciated not of suitable type - needs integer"
                        )

            except IndexError:
                raise ValueError(
                    "If X is a 2D array, must pass the relevant column index as hex_idx=... on init"
                )

        return self.get_even_clusters(latlongs, self.n_splits)

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray] = None,
        groups: Union[pd.Series, np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        :param X: array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        :type X: Union[pd.DataFrame, np.ndarray]
        :param y: array-like of shape (n_samples,),
            The target variable for supervised learning problems, defaults to None
        :type y: Union[pd.Series, np.ndarray], optional
        :param groups: Spatial group labels for the samples used while splitting the dataset into
            train/test set, defaults to None
        :type groups: Union[pd.Series, np.ndarray], optional
        :return: Generator of tuples of train and test indices
        :rtype: Iterator[Tuple[np.ndarray, np.ndarray]]
        :yield: Next set of train, test indices
        :rtype: Iterator[Tuple[np.ndarray, np.ndarray]]
        """
        return super().split(X, y, groups)


class StratifiedIntervalKFold(StratifiedKFold):
    """NB lightly edited version of stratified KFold
    - difference is just that class labels are generated using pd cut to make n_cuts even intervals
    (to improve folds w inflated vals), rather than just using values themselves"""

    def __init__(self, n_splits=5, *, n_cuts=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.n_cuts = n_cuts

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )

        y = column_or_1d(y)

        y_encoded = pd.cut(y, self.n_cuts)

        n_classes = self.n_cuts
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        if self.n_splits > min_groups:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (min_groups, self.n_splits),
                UserWarning,
            )

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype="i")
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds


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
