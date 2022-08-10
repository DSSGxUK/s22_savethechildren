import tensorflow
from tensorflow import keras
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
import numpy as np
from tensorflow.keras.optimizers import Adam
import gc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import keras_tuner as kt
from pyDRMetrics.pyDRMetrics import *
from stc_unicef_cpi.data import cv_loaders as cvl
from sklearn.preprocessing import StandardScaler
from stc_unicef_cpi.data import process_geotiff as pg
from pathlib import Path
import glob

random_state=0
np.random.seed(random_state)

def get_train_data(tiff_dir, hex_codes, dim=16):

  """
  Get training data for the autoencoder
  
  Inputs:
    tiff_dir: directory containing all country tiffs; function assumes tiffs have been clipped/reprojected where necessary
    hex_codes: H3 hexagons for training set
    dim: dimension for images extracted from rasters
  
  Outputs: numpy array of size (len(hex_codes), 16, 16, total channels from tiff_dir)
    
  """
  data = pg.convert_tiffs_to_image_dataset(tiff_dir, hex_codes, dim, dim)
  shape = data.shape

  # scale data
  data = StandardScaler().fit_transform(data.reshape(data.shape[0], -1))
  data = data.reshape(shape)

  # channels last
  data = np.transpose(data, (0, 2, 3, 1))

  # fill nans
  if np.isnan(np.sum(data)):
    lower_bound = np.min(data[~np.isnan(data)])
    if lower_bound < 0:
      data = np.nan_to_num(data, nan=2*lower_bound)
    else:
      data = np.nan_to_num(data, nan=-2*lower_bound)
  
  return data

def get_best_hyperparameters(input_data, random_state=0, validation_split=0.1,
                            batch_size=[64, 128],
                            learning_rate=[1e-2, 5e-3, 1e-3],
                            epochs=100, logdir="autoencoder",
                            project_name="tune_model", es_patience=5):
  
  """
  Get the tuned hyperparameters for the model

  Inputs:
    input_data: image input of size (num of samples, width, height, bands);
                reshaped & imputed input of convert_tiffs_to_image_dataset
    validation_split: split ratio for model.fit()
    random_state: random state
    batch_size: list of batch sizes to check
    learning_rate: list of learning rates to check
    epochs: max epochs for training; specify value slightly higher than expected convergence
    logdir: directory for logging
    project_name: name of project
    es_patience: patience for early stopping

  Outputs: dictionary with best learning rate, batch size & epoch
  """
  
  input_dims = input_data.shape

  class HyperModel(kt.HyperModel):
    def build(self, hp):
      
      model = tensorflow.keras.Sequential()
      
      model.add(keras.Input(shape=input_dims[1:]))
      model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'))
      model.add(MaxPooling2D((2, 2), padding='same', name='mp1'))
      model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2'))
      model.add(MaxPooling2D((2, 2), padding='same', name='mp2'))
      model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv3'))
      model.add(MaxPooling2D((2, 2), padding='same', name='mp3'))
      model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv4'))
      model.add(MaxPooling2D((1, 1), padding='same', name='Encoder_Output'))
      
      model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv5'))
      model.add(UpSampling2D((1, 1), name='us1'))
      model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv6'))
      model.add(UpSampling2D((2, 2), name='us2'))
      model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv7'))
      model.add(UpSampling2D((2, 2), name='us3'))
      model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv8'))
      model.add(UpSampling2D((2, 2), name='us4'))
      model.add(Conv2D(input_dims[-1], (3, 3), activation=None, padding='same', name='Decoder_Output'))
      
      hp_learning_rate = hp.Choice('learning_rate', values=learning_rate)
      
      model.compile(
          optimizer=Adam(learning_rate=hp_learning_rate),
          loss=keras.losses.MeanSquaredError()
          )
      print(model.summary())
      return model
    
    def fit(self, hp, model, *args, **kwargs):
      return model.fit(*args,
                     batch_size=hp.Choice("batch_size", batch_size),
                     validation_split=validation_split,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=es_patience)]#, TensorBoard("/tmp/tb_logs")]
                     )
  
  tuner = kt.Hyperband(HyperModel(),
                     objective='val_loss',
                     factor=5,
                     overwrite=True,
                     directory=logdir,
                     project_name=project_name,
                     seed=random_state)

  tuner.search(input_data, input_data)

  best_hps = tuner.get_best_hyperparameters()[0]
  print(f"""
  Best LR: {best_hps.get('learning_rate')}
  Best Batch Size: {best_hps.get('batch_size')}.
  """)
  
  model = tuner.hypermodel.build(best_hps)
  history = model.fit(input_data, input_data, epochs=epochs,
                      validation_split=validation_split,
                     )
  
  val_loss_per_epoch = history.history['val_loss']
  best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
  print('Best epoch: ', str(best_epoch))

  hps = {
      'learning rate': best_hps.get('learning_rate'),
      'batch size': best_hps.get('batch_size'),
      'epochs': best_epoch
  }
  return hps


def get_trained_autoencoder(input_data, batch_size=128, epochs=100, learning_rate=0.001, save_dir=None, model_name=None):
  
  """
  Get the trained model using tuned hyperparameters

  Inputs:
    input_data: image input of size (num of samples, width, height, bands);
                reshaped & imputed input of convert_tiffs_to_image_dataset
    batch_size): batch size for training
    epochs: number of epochs for training
    learning_rate: learning_rate for Adam optimizer
    save_dir: directory to save model in
    model_name: name of saved h5 model file

  Outputs: If save_dir=None, Keras sequential model else None
  """
  
  input_dims = input_data.shape

  model = tensorflow.keras.Sequential()
  
  model.add(keras.Input(shape=input_dims[1:]))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'))
  model.add(MaxPooling2D((2, 2), padding='same', name='mp1'))
  model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2'))
  model.add(MaxPooling2D((2, 2), padding='same', name='mp2'))
  model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv3'))
  model.add(MaxPooling2D((2, 2), padding='same', name='mp3'))
  model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv4'))
  model.add(MaxPooling2D((1, 1), padding='same', name='Encoder_Output'))
      
  model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv5'))
  model.add(UpSampling2D((1, 1), name='us1'))
  model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv6'))
  model.add(UpSampling2D((2, 2), name='us2'))
  model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv7'))
  model.add(UpSampling2D((2, 2), name='us3'))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv8'))
  model.add(UpSampling2D((2, 2), name='us4'))
  model.add(Conv2D(input_dims[-1], (3, 3), activation=None, padding='same', name='Decoder_Output'))
     
  model.compile(
          optimizer=Adam(learning_rate=learning_rate),
          loss=keras.losses.MeanSquaredError()
          )
  
  model.fit(input_data, input_data,
            batch_size=batch_size,
            epochs=epochs)
  
  if save_dir is not None:
      if model_name is None:
          print("Warning: No model_name given, saving as autoencoder.h5")
          model_name = "autoencoder"
      model.save(Path(save_dir)/(model_name+".h5"))
  else:
      return model


def get_encoded_features(trained_autoencoder_dir, model_name, hex_codes, tiff_files_dir, dim=16, batch_size=4096):

  """
  Get encoded features in batches

  Inputs:
    trained_autoencoder_dir: directory for saved keras model
    model_name: name of saved model inside trained_autoencoder_dir
    hex_codes: numpy array containing H3 hexagons for which to generate predictions
    tiff_files_dir: file path for geotiffs
    dim: dimension for images extracted from rasters
    batch_size: batch size for getting predictions

  Outputs: numpy array of size (len(hex_codes), 32)
  """
  
  trained_autoencoder = tensorflow.keras.models.load_model(Path(trained_autoencoder_dir) / (model_name+".h5"))
  
  intermediate_model = tensorflow.keras.models.Model(inputs=trained_autoencoder.input,
                                                     outputs=trained_autoencoder.get_layer('Encoder_Output').output)
  
  files = glob.glob(str(Path(tiff_files_dir)/"*.tif"))
  params = {'dim': (dim, dim),
           'hex_idxs': hex_codes,
           'batch_size': batch_size,
           'data_files': files,
           'shuffle': False}
  
  pred_generator = cvl.KerasDataGenerator(**params)

  encodings = intermediate_model.predict(pred_generator, use_multiprocessing=True, workers=6)

  return encodings.reshape(encodings.shape[0], -1)


def check_autoencoder_reconstruction(trained_autoencoder, input_data):
  
  """
  Plot reconstructed images to check performance of autoencoder

  Inputs:
    trained_autoencoder: output of get_trained_autoencoder() or saved keras model
    input_data: image input of size (num of samples, width, height, bands);
                reshaped & imputed input of convert_tiffs_to_image_dataset

  Outputs: None
  """
  decoded_imgs = trained_autoencoder.predict(input_data)
  n = 5
  plt.figure(figsize=(20, 8))
  for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    band = np.random.randint(0, input_data.shape[-1])
    sample = np.random.randint(0, input_data.shape[0])
    plt.imshow(input_data[sample,:,:,band].reshape(input_data.shape[1], input_data.shape[1]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Actual Sample:"+str(sample)+", Band:"+str(band))

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[sample,:,:,band].reshape(input_data.shape[1], input_data.shape[1]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Reconstructed Sample:"+str(sample)+", Band:"+str(band))
  
  plt.show()

def get_encoding_metrics(original_data, encoded_features):

  """
  Get residual variance, AUC Trustworthiness & AUC Continuity for encodings

  Inputs:
    original_data: image input of size (n_samples, width, height, bands) or (n_samples, width*height*bands)
    encoded_features: encoded features of size (n_samples, 32) or (n_samples, 2, 2, 8)

  Outputs: None
  """
  
  if original_data.ndim > 2:
    original_data = original_data.reshape(original_data.shape[0], -1)
  
  if encoded_features.ndim > 2:
    encoded_features = encoded_features.reshape(encoded_features.shape[0], -1)

  drm = DRMetrics(original_data, encoded_features)
  print("Residual Variance: ", str(drm.Vrs))
  print("AUC Trustworthiness: ", str(drm.AUC_T))
  print("AUC Continuity: ", str(drm.AUC_C))