import pandas as pd
import rasterio
import glob
import numpy as np
import tensorflow as tf
import keras
from keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import itertools
from sklearn.preprocessing import StandardScaler

def numbands_from_tiffs(dir):
    """
    Get the name and number of bands for each tiff

    Inputs: tiff directory (string)
    Outputs: 2 equal size lists containing bands per tiff and tiff name 
    """
    num_bands = []
    band_names = []
    path = dir.rstrip("/") + "/*.tif"
    tif_files = glob.glob(path)
    for tif in tif_files:
        band_names.append(tif.rpartition('cpi')[2][:-8])
        with rasterio.open(tif) as open_file:
            num_bands.append(len(open_file.indexes))
    return num_bands, band_names

def get_features(img_arr, num_bands, pca_components, dim=32):
    """
    For each raster, get top PCA features using
    2048 features from pretrained ResNet50

    Inputs: 
        img_arr: output of convert_tiffs_to_image_dataset
        num_bands: first output of numbands_from_tiffs
        pca_components: number of components to reduce features to
        dim: dimensions for ResNet50 input; should match shape[2] and shape[3] of img_arr
    Outputs:
        array features of size (number of samples: img_arr.shape[0],
                                num_features: number of 3-groupings obtained from tiffs,
                                pca_components: specified as funtion arg)
    """

    num_features = sum([j//3 + 1 if j%3 else 0 for j in num_bands])
    features = np.zeros((img_arr.shape[0], num_features, pca_components))
    
    n_feature = 0

    for i in range(len(num_bands)):

        # get bands for each tif

        if i == 0:
            data = img_arr[:, 0:num_bands[i], :, :]
        elif i == len(num_bands)-1:
            data = img_arr[:, -num_bands[i]:, :, :]
        else:
            data = img_arr[:, sum(num_bands[:i]):sum(num_bands[:i])+num_bands[i], :, :]
        
        data = data.reshape(data.shape[0], data.shape[2], data.shape[3], data.shape[1])
        
        # case where tif has 1 or 2 bands

        if data.shape[3] < 3:
            npad = ((0, 0), (0, 0), (0, 0), (0, 3-data.shape[3]))
            data_pad = np.pad(data, npad, 'constant')

            model = ResNet50(include_top=False, weights='imagenet', input_shape=(dim, dim, 3))
            resnet_features = model.predict(preprocess_input(data_pad)).reshape(data.shape[0], 2048)
            
            scaler = StandardScaler()
            pca = PCA(n_components=pca_components)
            pca_features = pca.fit_transform(scaler.fit_transform(resnet_features))
            
            features[:,n_feature,:] = pca_features
            n_feature += 1
        
        #case where tif has more than 3 bands
        elif data.shape[3] > 3:

            #sub_bands = data.shape[3]//3 + 1 if data.shape[3]%3 else 0
            for j in range(0, data.shape[3], 3):
                if j == (data.shape[3]//3)*3:
                    sub_data = data[:,:,:,j:]
                else:
                    sub_data = data[:,:,:,j:j+3]
                
                if sub_data.shape[3] < 3: # change to less than
                    npad = ((0, 0), (0, 0), (0, 0), (0, 3-sub_data.shape[3]))
                    sub_data = np.pad(sub_data, npad, 'constant')
                
                model = ResNet50(include_top=False, weights='imagenet', input_shape=(dim, dim, 3))
                resnet_features = model.predict(preprocess_input(sub_data)).reshape(data.shape[0], 2048)
                
                scaler = StandardScaler()
                pca = PCA(n_components=pca_components)
                pca_features = pca.fit_transform(scaler.fit_transform(resnet_features))
                
                features[:,n_feature,:] = pca_features

                n_feature += 1
        
        #case where tif has exactly 3 bands; not required now but adding in case needed later
        else:

            model = ResNet50(include_top=False, weights='imagenet', input_shape=(dim, dim, 3))
            resnet_features = model.predict(preprocess_input(data)).reshape(data.shape[0], 2048)
            
            scaler = StandardScaler()
            pca = PCA(n_components=pca_components)
            pca_features = pca.fit_transform(scaler.fit_transform(resnet_features))
            
            features[:,n_feature,:] = pca_features
            n_feature += 1
        
    return features